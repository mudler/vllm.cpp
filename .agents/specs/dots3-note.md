# dots3-note — a DeepSeek-V3.2 text tower we mostly own, on hardware we do not

**Rows:** `MODEL-MM-dots3-note-dots3-note-for-causal-lm` (`Dots3NoteForCausalLM`),
`MODEL-SPEC-dots3-note-dots3-note-mtp` (`Dots3NoteMTPModel`) — both in
[model-matrix.md](../model-matrix.md).
**Issue:** [#699](https://github.com/mudler/vllm.cpp/issues/699).
**Claim:** `CLAIM-MODEL-DOTS3-NOTE-W0`.
**Checkpoint:** `dots-studio/dots3-note-prev` (bf16) and
`dots-studio/dots3-note-prev-fp8`.
**Upstream:** vLLM `main` — added by
[vllm#51255](https://github.com/vllm-project/vllm/pull/51255) at
`9035151d6`, last touched `170592a93` (2026-08-13,
[vllm#52172](https://github.com/vllm-project/vllm/pull/52172) "Disable sequence
parallelism for Dots3 NOTE"). **NOT present at our parity pin.**
**Designated CUDA host (developer directive, 2026-08-14):** Thor, reached as the
fleet device **`thor:gpu0` through an `rc` lease and never by `ssh`** — the host
address is recorded in `environment.md` to identify the box, not as a way into
it. §6.3 records what that host can and cannot carry for this model, measured.
**Status:** W3 — the FULL-attention layer landed as a portable host reference
(§7 W3, evidence §4.5), on top of W2's whole weight map (§4.4) and W1's config +
registry (§4.1). The arch RESOLVES, parses, accounts for 38006/38006 of the
released checkpoint's tensors, and now COMPUTES `_forward_note_mla`'s full arm
against an independent double-precision reference. Load, GGUF and the DEVICE
forward still REFUSE BY NAME, and the reference is deliberately not on the
decode path — see `## Owed`. No GPU has been used at any brick and no tensor
byte of the checkpoint has been downloaded: the committed fixtures are the
released `config.json` and a headers-only projection of the complete shard
index. The row stays `SPIKE`.

---

## 0. Honesty statement — what is and is not claimed

Nothing has been measured on this model, and one thing has now been ported: W3
landed `_forward_note_mla`'s full-attention arm as a portable host reference
(§4.5). It computes; it is not reachable from the decode path and it has never
been compared against vLLM, because vLLM cannot run this model on any host this
project owns (§6.2). "Ported" here therefore means "written from the upstream
source and agreed with a second independent implementation of the same
formula" — never "matches the oracle".

**Measured here:** the checkpoint's file list and byte total (HF API,
`?blobs=true`); the geometry in §1 (its `config.json`); the presence and shape of
the upstream implementation (`git show origin/main:...` against a fetched
`${VLLM_SOURCE}`); the absence of the architecture at our pin; and the live state
of Thor in §6.3 (one read-only `ssh` probe).

**Read, not run:** every upstream behaviour in §2. No vLLM execution of this
model has happened on this project's hardware, and §6 explains why it cannot.

**Not established, and this is the load-bearing gap:** that any oracle for this
model runs anywhere we can reach. Until it does, no gate in §5 can bind, and no
brick past W1 may claim correctness against upstream — only against an
independent in-test reference.

A second honesty note, because this row's shape invites the error. The text tower
is *derived from* code we have gated (DeepSeek-V2 MLA, the V3.2-family DSA
indexer, the `noaux_tc` router). Derived is not identical. §4 lists four config
fields and one norm that differ from the DeepSeek defaults our code assumes; each
one silently changes numerics rather than failing loudly.

---

## 1. What the model is, measured

`dots-studio/dots3-note-prev`, `model_type = "dots3_note"`, architecture
`Dots3NoteForCausalLM`. 280B total / 16B activated (model card), text + image +
video + audio understanding, 512K positions.

**Checkpoint, from the HF API:** 131 language shards + `model-vision.safetensors`
(13.7 GB) + `model-audio.safetensors` (1.77 GB) = **~576 GB** repo total in bf16.
The `-fp8` sibling is ~290 GB. There is no smaller `dots3-note` variant in the
`dots-studio` org — the org's other models (`dots.llm1`, `dots.ocr`, `dots.mocr`,
the `dots.tts*` family) are different architectures, not scaled-down NOTEs.

### 1.1 Text tower

| Field | Value |
|---|---|
| `num_hidden_layers` | 46 |
| `hidden_size` | 5120, `intermediate_size` 13824 |
| `vocab_size` | 152064, `tie_word_embeddings` false |
| `max_position_embeddings` | 524288, `rope_theta` 8e7, `rope_scaling` null |
| MoE | 256 routed + 1 shared, `num_experts_per_tok` 8, `moe_intermediate_size` 1536, `moe_layer_freq` 1, `first_k_dense_replace` 1 |
| Router | `scoring_func` sigmoid, `topk_method` noaux_tc, `norm_topk_prob` true, `routed_scaling_factor` 1.0 |

**Attention is hybrid, and the two halves have different geometry.**
`layer_types` is 46 entries: **13 `full_attention`** at indices
`0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45`, and **33 `sliding_attention`**
filling the rest (period 4 after the first pair).

| | full layers | sliding layers |
|---|---|---|
| heads | 128 | 64 (`swa_num_attention_heads`) |
| `q_lora_rank` | 1024 | 1024 (`swa_q_lora_rank`) |
| `kv_lora_rank` | 512 | **1024** (`swa_kv_lora_rank`) |
| `qk_nope_head_dim` | 128 | **192** (`swa_qk_nope_head_dim`) |
| `qk_rope_head_dim` | 64 | 64 |
| `v_head_dim` | 128 | 128 |
| rope | theta 8e7, GPT-J | **theta 5e4** (`swa_rope_theta`), GPT-J |
| rope layout | \*\*both\*\* geometries are `is_neox_style=False` — see §4 item 6 and [#1804](https://github.com/mudler/vllm.cpp/issues/1804) | |
| window | — | **513** (`sliding_window_size`) |
| gate | `headwise` | `headwise` (`swa_attention_gate_type`) |
| sparse indexer | yes | no |

Consequence for the KV cache: the full layers want a latent row of
`512 + 64 = 576`, the sliding layers `1024 + 64 = 1088`. Upstream reconciles this
by *padding the physical row* so both share one block shape (§2.3).

**DSA (lightning indexer)** on the full layers: `index_n_heads` 64,
`index_head_dim` 128, `index_topk` 2048.

### 1.2 Vision tower — a MoE ViT

`vision_config`: 42 layers, `embed_dim` 1536, `intermediate_size` 4224,
`moe_intermediate_size` 2112, 24 heads, `patch_size` 14, `temporal_patch_size` 1,
`spatial_merge_size` 2, `use_qk_norm` true, `is_causal` false, `post_norm` true,
`pre_pixel_shuffle` true, `use_bias` false. Adapter is `patch_merger`,
1536 → 5120, merge size 2.

The MoE is a **pyramid**: `pyramid_num_routed` is `-1` (dense) for layers 0–24,
then `4, 8, 12, … 64, 64` for layers 25–41. Router `sigmoid`, `router_scale` 1.0,
`capacity_factor` 2. ~7B total / 1.2B activated (model card).

### 1.3 Audio tower

`audio_config.encoder_type = "dots"`, wrapping a modified Whisper: `d_model`
1280, 32 encoder layers, 20 heads, `encoder_ffn_dim` 5120, 128 mel bins,
`max_source_positions` **6000**, `activation_function` **swiglu**. Plus
`use_conv2d_stem` true, `use_rope` true (partial rotary 0.5, theta 1e4),
`use_rms_norm` true, `use_causal` false, `downsample_hidden_size` 480,
`chunk_seconds` 60, `conv_chunksize` 500, adapter 1280 → 5120, 16 kHz. ~800M
(model card). Audio placeholders are `<|audio_comp_start|>` /
`<|audio_comp_pad|>` / `<|audio_comp_end|>`.

### 1.4 MTP

`Dots3NoteMTPModel`. The config class defaults `num_nextn_predict_layers` to
**1** — one nextn layer, not three. The model card's "three-token speculative
decoding" therefore describes the *speculation depth* the head is driven at, not
a count of heads.

**W0 wrote that reading as inferred from the config default and NOT verified
against a checkpoint, owing the answer to W2. W1 read the shard index and
RESOLVED it**: the release carries backbone layers 0-45 and exactly one more,
`model.layers.46.*` (18 tensors), plus `model.mtp.embed_tokens.weight`. The
reading holds. W1 also measured two things about that block that upstream does
not state — it carries the SLIDING attention tensor set and a DENSE MLP — and
§4.1 records both, together with the reconciliation W10 owes because
`config.layer_types` has no entry at index 46.

---

## 2. Upstream chain, `file:line`

Paths under `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` at `origin/main`. Note
the package layout: this is **not** `vllm/model_executor/models/dots3_note.py`.
It is `vllm/models/dots3_note/{common,nvidia}/`, the same platform-split shape
DeepSeek-V4 uses — 15 files, ~5.7k LoC.

### 2.1 Registration

- `vllm/model_executor/models/registry.py:381` (`_MULTIMODAL_MODELS`) —
  `"Dots3NoteForCausalLM": ("vllm.models.dots3_note", "Dots3NoteForCausalLM")`.
- `registry.py:670` (`_SPECULATIVE_DECODING_MODELS`) — `"Dots3NoteMTPModel": ("vllm.models.dots3_note", "Dots3NoteMTP")`.
  Both re-derived at `c205726108df54bb6fbf15b19e725a4a3add2b18`; W0 recorded
  `:375` and `:662`, which is where they sat at the revision W0 read.
- `vllm/transformers_utils/configs/dots3_note.py:7` —
  `class Dots3NoteConfig(DeepseekV3Config)`. Read this file before anything else;
  §4 is entirely about what it sets.

### 2.2 The text tower is subclassed from DeepSeek

`vllm/models/dots3_note/nvidia/model.py`:

| dots3 class | base | line |
|---|---|---|
| `Dots3NoteMoE` | `DeepseekV2MoE` | `:76` |
| `Dots3NotePaddedMLAAttention` | `MLAAttention` | `:204` |
| `Dots3NoteFullAttention` | `DeepseekV2MLAAttention` | `:219` |
| `Dots3NoteSlidingAttention` | `nn.Module` (built directly) | `:329` |
| `Dots3NoteDecoderLayer` | `DeepseekV32DecoderLayer` | `:481` |
| `Dots3NoteModel` | `DeepseekV32Model` | `:549` |
| `Dots3NoteLanguageModelForCausalLM` | `DeepseekV32ForCausalLM` | `:681` |

The shared attention body is the free function `_forward_note_mla`
(`model.py:135`), used by both the full and the sliding classes. Its deltas over
plain DeepSeek MLA are exactly four:

1. `q_c = q_a_layernorm(q_c) * q_lora_scale` and
   `kv_c_normed = kv_a_layernorm(kv_c) * kv_lora_scale` (`:155`, `:159`) — see §4.2.
2. `k_pe = k_rope_only_layernorm(k_pe)` (`:160`) — an **extra RMSNorm over the
   64-dim rope-only slice of k**, which DeepSeek does not have.
3. The headwise gate (`:190-197`): `gate = g_proj(hidden_states)`,
   `sigmoid` in **fp32** then cast back, reshape attention output to
   `[-1, num_heads, v_head_dim]`, multiply per head, flatten.
4. The indexer runs only when `attention.is_sparse` (`:171`) — i.e. never on the
   sliding layers, which set `self.indexer = None` / `is_sparse = False`
   (`model.py:432-434`).

**ALL FOUR items in this list carried W0-era line citations, every one of them
stale, and they are corrected in place** (as W1 corrected §4 item 6 — `main` is never rewritten,
so a wrong anchor is repaired where a reader will hit it). W0 read
`9035151d6`/`170592a93`; re-derived at `06ecec7a84`, where
`git log 185cada36b..06ecec7a84 -- vllm/models/dots3_note/` is EMPTY, the gate
is `:190-197` and not `:246-262` (which is a constructor argument list), the
`is_sparse` guard is `:171` and not `:186`, and the sliding class's three
assignments are `:432-434` and not `:430-432`. Items 1 and 2 were stale too and
by a smaller margin, which is the more dangerous kind: the two rescales are
`:155` and `:159` rather than `:154` and `:160`, and `k_rope_only_layernorm` is
`:160` rather than `:161` — so W0's citation for the rescale landed exactly on
the line the NEXT item names, and a reader checking it would have found
plausible code and moved on. An earlier draft of this very paragraph said items
1 and 2 were "verified unchanged", which was the same mistake one level up and
is corrected here rather than quietly dropped.
The sources are byte-identical between the two revisions, so this is
pure anchor rot rather than an upstream change — which is exactly what
`check-symbol-anchors.py` cannot catch, because its own docstring says it does
not verify line citations
([#1139](https://github.com/mudler/vllm.cpp/issues/1139)).

`Dots3NoteMoE` (`:76`) is `DeepseekV2MoE` with the shared expert lifted out and
rebuilt at a **block-padded** intermediate size (`_padded_mlp_size:63`) so
blockwise-FP8 weights divide across TP ranks. At TP=1 with no
`weight_block_size` this padding is the identity — record it, do not port it
speculatively.

### 2.3 Sliding-window MLA — the new machinery

`vllm/models/dots3_note/nvidia/attention.py`, 807 lines, and the reason this row
is not a mechanical port:

- `_gather_swa_kv_kernel:48` — Triton kernel gathering the windowed KV rows.
- `_apply_swa_score_mask_kernel:118` — the window mask over scores.
- `_build_sliding_window_metadata:191`, `_SlidingWindowChunk:171`,
  `_SlidingWindowMetadata:186` — chunked window bookkeeping.
- `Dots3NoteMLAMetadataBuilder(TritonMLAMetadataBuilder):306`, including
  `_reserve_attn_logits_workspace:315`.
- `Dots3NoteTritonMLAImpl(TritonMLAImpl):438` with `_forward_swa_mqa:469`,
  `forward_mha:564`, `forward_mqa:655`.
- `Dots3NoteFlashAttnPrefillBackend:257` with `run_sliding_window:278`.
- `Dots3NotePaddedSparseImpl(FlashAttnMLASparseImpl):689` — the full layers,
  with `_logical_cache:692` narrowing the padded physical row back to the logical
  576 and `do_kv_cache_update:696`.

The padding contract lives in `model.py:204-217`: `Dots3NotePaddedMLAAttention`
overrides `get_kv_cache_spec` to report `physical_head_size`, so full and sliding
layers allocate the same block while each reads its own logical width.

### 2.4 Vision

`nvidia/vision.py` (677) — `DotsMoEVitConfig:27`, `MoESwiGLUFFN:139`,
`MoESwiGLUFFNFP8:242`, `DotsPatchEmbed:321`, `MoEVisionBlock:352`,
`PixelShuffleAdapter:419`, `PatchMergerAdapter:464`, `DotsMoEVitModel:508`
(`get_pos_ids_by_grid:565`, `rot_pos_emb:601`, cu_seqlens builders `:609`/`:621`).
`nvidia/vision_attention.py` (477). `nvidia/vision_moe.py` (149) —
`note_vision_fused_moe_fp8`, whose docstring states the memory-format trap
directly: *"the native NOTE encoder keeps dynamic activation scales as FP32
instead of rounding them to E8M0"*, weight scales block 128×128, activations
quantized per token/group-128.

### 2.5 Audio

`nvidia/audio_encoder.py` (745) — `RotaryEmbedding:47`,
`WhisperPositionalEmbedding:184`, `WhisperAttention:196`,
`WhisperEncoderLayer:310`, `DotsSpeechEncoder:437` with three stems
(`_forward_conv2d_stem:573`, `_forward_conv1d_stem:594`,
`_forward_latent_stem:607`) and `_temporal_mask:538`. `nvidia/audio.py` (305) is
the vLLM-side wrapper.

### 2.6 Multimodal front end and MTP

`nvidia/multimodal.py:49` `Dots3NoteForCausalLM(nn.Module, SupportsMultiModal,
SupportsPP)` — `get_placeholder_str:65`, `_process_image_input:144`,
`_process_audio_input:156`, `_process_video_input:172`, `embed_multimodal:225`,
`get_mm_mapping:300`. `common/processor.py` (811) and `common/video.py` (497)
carry the prompt-side expansion and frame sampling.
`nvidia/mtp.py:31,88,141` — `Dots3NoteMultiTokenPredictorLayer`,
`Dots3NoteMultiTokenPredictor(DeepseekV32MultiTokenPredictor)`,
`Dots3NoteMTP(DeepseekV32MTP)` with `has_own_embed_tokens = True`,
`has_own_lm_head = False`.

---

## 3. Reuse versus new

### 3.1 What we already own and gate

| Piece | Ours | Evidence |
|---|---|---|
| MLA: fused qkv_a, q_lora branch, two RMSNorms, split RoPE, kv_b | `src/vllm/model_executor/models/deepseek_v2.cpp` | SACRED 8/8 token-exact on DeepSeek-V2-Lite |
| DSA lightning indexer + top-k | `deepseek_v4_dsa.cpp`, [dsa-topk-bounds.md](dsa-topk-bounds.md) | unit-gated |
| `noaux_tc` sigmoid router + `e_score_correction_bias` + shared experts | `deepseek_v2`, `laguna_ops.cpp`, `kimi_k3.cpp`, `nemotron_h_weights.cpp:309` | `tests/vt/test_ops_moe_router_grouped.cpp` |
| 256-expert grouped-GEMM MoE | DeepSeek-V4 MoE path | measured decode |
| MTP draft head + lossless self-spec verify | `v1/worker/gpu/spec_decode/mtp/speculator.cpp`, `deepseek_v4.cpp` | `test_deepseek_v4_mtp` 5/5 |
| ViT + 2-D RoPE + patch merger, **image and video** | `qwen3_vl_vision.cpp` | STRICT 32/32 image, 32/32 video |
| Whisper-class audio encoder + FA-2 attention | `whisper_audio.cpp`, `voxtral.cpp` | audio→text gate 16/16 |
| MM processor / placeholder expansion / OpenAI content parts | `multimodal/`, `chat_mm.cpp` | `test_chat_mm` 8/8 |

That is the majority of the parameter count and most of the decode step.

### 3.2 What does not exist here

1. **Windowed MLA.** Every MLA path we have is full-attention over a paged cache
   (`deepseek_v4_registry.cpp` says so explicitly: `is_hybrid = false`). 33 of 46
   layers need a 513-wide window. **Largest single brick.**
2. **Heterogeneous MLA KV spec** — two logical latent widths sharing one physical
   block, plus a `_logical_cache`-equivalent narrowing on read.
3. **Headwise attention gate**, the extra `k_rope_only_layernorm`, and the two
   lora rescale scalars. Individually trivial, all on the hot path, and all three
   are invisible to a shape check.
4. **MoE ViT** — every vision tower we have is dense. Needs the pyramid schedule,
   the sigmoid router with `capacity_factor`, and the FP32-scale FP8 MoE formula
   of §2.4.
5. **`dots` audio stem** — conv2d stem, RoPE, RMSNorm, SwiGLU, 6000 positions,
   60 s chunking. Our Whisper block has none of these.
6. **`dots3_note` config parsing**, including the four defaults of §4 that the
   checkpoint's `config.json` does **not** carry. **LANDED at W1, and NOT in
   `hf_config.cpp` as this item first said — see §4.2 for why.**
7. **GGUF k-quant arm.** llama.cpp has no `dots3_note` architecture, so the
   converter is ours to write and there is no quant-matched llama.cpp
   comparison for this row. Per AGENTS.md the arm is owed, not optional; an
   unimplemented arm refuses with a message naming the missing piece.

---

## 4. Config traps — what `config.json` does not say

`Dots3NoteConfig.__init__` (`transformers_utils/configs/dots3_note.py:12-25`)
sets four defaults that are **absent from the published `config.json`**. A port
that reads only the checkpoint gets all four wrong, and every one of them is
numerically silent.

1. **`n_group = 1`, `topk_group = 1`.** Upstream's comment is explicit: *"Do not
   inherit DeepSeek-V3's 8-group/4-group router defaults: Note was trained with
   an ungrouped (1/1) noaux_tc router … A different grouping changes the selected
   experts at every MoE layer."* Our `noaux_tc` router is gated at V3's grouped
   dims; this row must drive it ungrouped.
2. **`indexer_rope_interleave = True`.** dots3 projects indexer RoPE coordinates
   in **adjacent (GPT-J) pairs**; DeepSeek-V3.2 defaults to **split-half (NeoX)**
   when the flag is absent. Our DSA indexer was ported against the V3.2 default,
   so this rotates different learned coordinates.
3. **`num_nextn_predict_layers = 1`** (§1.4).
4. The base class is `DeepseekV3Config`, so anything not overridden inherits
   V3's default — check each field we read rather than assuming the JSON is
   complete.

Two more that *are* in the JSON but differ from our assumptions:

5. **`apply_mla_qkv_lora_rescale: true`** ⇒
   `q_lora_scale = sqrt(hidden_size / q_lora_rank)` and
   `kv_lora_scale = sqrt(hidden_size / kv_lora_rank)` applied *after* the
   respective layernorms (`model.py:154,160`). For the full layers that is
   `sqrt(5120/1024)` and `sqrt(5120/512)`; for sliding, `sqrt(5120/1024)` twice.
6. **`swa_rope_theta = 5e4` on the sliding layers**, against the full layers'
   `rope_theta = 8e7` (`model.py:404-407`). Three orders of magnitude apart, on
   33 of the 46 layers.

   **W0 wrote this item as "`is_neox_style=False` on the sliding rope only",
   and that half is WRONG — corrected in place at W1
   ([#1804](https://github.com/mudler/vllm.cpp/issues/1804)), because `main` is
   never rewritten.** It is not sliding-only. `Dots3NoteSlidingAttention` does
   pass `is_neox_style=False` literally (`model.py:408`), and
   `Dots3NoteFullAttention` inherits the SAME hard-coded value from
   `deepseek_v2.py`::`DeepseekV2MLAAttention.__init__` (`:1093-1098`). **BOTH
   MLA ropes are GPT-J**, so the two geometries do not differ on the layout at
   all; they differ on the theta above. The sentence as written would have sent
   a W3 implementer to rotate the 13 full-attention layers split-half.

   The polarity that DOES flip is the **indexer's**, and that belongs to trap 2
   rather than here: `deepseek_v2.py:1148` sets the indexer rope to
   `is_neox_style = not indexer_rope_interleave`, so at DeepSeek-V3.2's
   absent-key default the indexer runs NeoX beside an MLA rope that is GPT-J,
   and `indexer_rope_interleave = True` is what makes dots3-note's two agree.
   Verified by RED-first assertion at W1, not by re-reading.

**Gate obligation:** each of the six gets a RED-first unit assertion before the
layer that consumes it is written. A wrong value here produces plausible tokens,
which is precisely the class of defect a token gate cannot catch when no oracle
is available to compare against.

### 4.1 W1 discharged the obligation — the RED-before evidence

**LANDED at W1** (`row/MODEL-MM-dots3-note-W1`,
`tests/vllm/models/test_dots3_note_scaffold.cpp`, upstream read at vLLM
`origin/main` `c205726108df54bb6fbf15b19e725a4a3add2b18`).

The RED arm was built and run BEFORE the correct values existed. It read the
traps the way a port that never opened `Dots3NoteConfig.__init__` would, and it
compiled — a mutation that fails to build reads as a passing test, so the red
result is a real run and not an inference. **The green arm is 19 cases /
3876 assertions.**

**The numbering below is §4's own**, 1 to 6 as the list above states them. An
earlier draft of this table split trap 1 into two and renumbered everything
after it; every citation in the code, the tests and the records now uses §4's
numbering, so a reader who follows "§4 trap 3" from a comment lands on the nextn
item and not on the indexer.

| Trap | RED reading | The assertion that fired | GREEN |
|---|---|---|---|
| 1 `n_group` / `topk_group` | 8 and 4 (DeepseekV3Config `:168-169`) | `CHECK( 8 == 1 )` — "n_group resolved to 8, not 1 — DeepSeek-V3's default of 8 regroups the router at every MoE layer"; `CHECK( 4 == 1 )` for `topk_group` | 1 and 1, and a grouped config is REFUSED by name |
| 2 `indexer_rope_interleave` | false (`deepseek_v2.py:1148` getattr default) | `CHECK( false )` — "indexer_rope_interleave resolved FALSE — that is DeepSeek-V3.2's absent-key default, not dots3-note's"; and `!indexer_rope_is_neox_style()` | true ⇒ GPT-J; an explicit `false` in JSON is still honoured |
| 3 `num_nextn_predict_layers` | 0 (absent key) | `CHECK( 0 == 1 )`, and the knock-on: "UNCLAIMED checkpoint tensors, first: model.layers.46.eh_proj.weight (19 total)" | 1, and the checkpoint agrees — see below |
| 4 field completeness | 36 required keys read with a SILENT fallback | 63 assertions across three cases, each naming its key: 26 absent keys parsed clean, 36 wrong-typed keys parsed clean, and a wrapped layout parsed to defaults | absent or wrong-typed REFUSES BY NAME — see §4.3 |
| 5 `apply_mla_qkv_lora_rescale` | never applied (our MLA has no scalar) | `REQUIRE( false )` on the flag, then the four scales | `sqrt(5120/1024)`, `sqrt(5120/512)`, and the sliding pair; full and sliding kv DISAGREE |
| 6 `swa_rope_theta` + layout | model-level theta; NeoX | `CHECK( 8e+07 == Approx( 50000 ) )`, "the two geometries resolved the SAME rope theta", and two `CHECK_FALSE( true )` on the layouts | 5e4 vs 8e7; both layouts GPT-J |

### 4.2 Where the config parsing lives — NOT `hf_config.cpp`

§3.2 item 6 named `hf_config.cpp`. W1 put it in the model's own TU
(`src/vllm/model_executor/models/dots3_note.cpp`) instead, and the reason is a
rule rather than a preference: AGENTS.md forbids "a surface that every PR must
write", and `hf_config.cpp` is the shared container reader every architecture
would otherwise edit. dots3-note contributes ~30 architecture scalars no other
model reads, which is exactly what `DeepseekV4Params`, `NemotronHParams` and
`MuseGlimmerParams` each keep in their own TU.

`hf_config.cpp` needs **no** edit for this checkpoint — measured, not assumed:
`LoadHfConfig` parses the released `config.json` unchanged. In particular it
must NOT normalize `sliding_window_size` into the typed `sliding_window`,
because upstream does not either: the window is handed to one `MLAAttention`
per sliding layer (`model.py:457`), never to the model-level config.

---

### 4.3 Trap 4 had no row until the review put one there

W1 landed traps 1, 2, 3, 5 and 6 with an assertion each and **left trap 4
ungated**, which is how it stayed the only item in §4 with nothing behind it.
The [#1805](https://github.com/mudler/vllm.cpp/pull/1805) review found the
consequence rather than the omission: `ParseDots3NoteParams` read every field
through a reader that substituted a fallback when the key was **absent**, and
the same fallback when the value had the **wrong JSON type**.

Deleting `apply_mla_qkv_lora_rescale` and `swa_rope_theta` from the fixture made
the parse SUCCEED, with all four LoRA scales at 1.0 and 33 of the 46 layers
rotating at 1e4 instead of 5e4. Neither key is one of the four
`Dots3NoteConfig.__init__` setdefaults; upstream reads both as plain attributes
and raises `AttributeError`. So W1 was not mirroring upstream here, it was
quietly more permissive than upstream, in the one direction §6.4 says nothing
can catch.

The blast radius measured wider than the two keys the review probed: **26 of the
36 required keys parsed clean when deleted**, and all 36 when given a wrong
type.

**What the fix does, and the line it draws.** A field this port reads is now
exactly one of two things:

- **Required.** Absent or wrong-typed refuses by name. 36 keys, every one of
  them carried by the released `config.json`.
- **Optional with an upstream-anchored default.** The four
  `configs/dots3_note.py` setdefaults, plus `moe_layer_freq` (`model.py:513`)
  and `routed_scaling_factor` (`model.py:546`), which really are `getattr`s, and
  `tie_word_embeddings`. A wrong TYPE still refuses: upstream would carry a
  string into arithmetic, not substitute its default.

For the subset of required keys that ARE `DeepseekV3Config` dataclass fields,
refusing is **stricter than upstream**, which would silently substitute V3's
default. That is deliberate, it is argued in the commit, and it is the same call
`hf_config.cpp` already makes for `output_gate_type`. The refusal message says
which upstream behaviour it stands in for — an `AttributeError`, a substituted
V3 default, or (for `index_topk`) a silent switch off the V3.2 sparse path
entirely — so the two cases stay distinguishable.

A **wrapped** config layout (`text_config`, `llm_config`, `thinker_config`) now
refuses too. dots3-note's released config is flat, and reading a wrapped one at
the top level would produce an all-defaults model with no error
(porting-a-model.md §1).

**One honest limit, recorded rather than papered over.** Eleven of the 36 keys
are also typed on the shared `HfConfig`, so `LoadHfConfig` refuses a wrong TYPE
first, with a message that names the config path and the JSON type instead of
the key. That is a real refusal from a shared component and the test asserts
THAT message for those eleven, rather than pretending dots3 caught them. All 36
still refuse; only the layer that refuses differs.

**§1.4 is RESOLVED, by the checkpoint rather than by inference.** The released
`model.safetensors.index.json` carries backbone layers 0-45 and **exactly one**
more, `model.layers.46.*` (18 tensors), plus `model.mtp.embed_tokens.weight`.
So `num_nextn_predict_layers = 1` is what the checkpoint ships, and the model
card's "three-token speculative decoding" is the depth ONE head is driven at.
`shared_head.head.weight` is absent, matching `has_own_lm_head = False`
(`mtp.py:142`).

**Two facts W1 measured that upstream does not state.** The nextn block carries
the **sliding** attention tensor set — no `indexer.*`, and `q_b_proj` is
[16384, 1024] = 64 x (192+64) — with a **dense** MLP. Upstream cannot answer the
first: `Dots3NoteDecoderLayer` selects its attention class from
`config.layer_types[layer_idx]` (`model.py:503`) and `layer_types` has no entry
at index 46. **W10 owes that reconciliation**; W1 takes the checkpoint as the
authority and says so at the enumeration site.

And a **memory-format** fact, per [`porting.md`](../porting.md): the whole
language tower is BF16 except one family. `mlp.gate.e_score_correction_bias`
ships **F32**. A loader that assumed one dtype for the checkpoint would misread
it, and no token gate could see the difference.

### 4.4 W2 read the whole index, and three things the slice could not say

**LANDED at W2** (`row/MODEL-MM-dots3-note-W2`, same TU and same test file as
W1, upstream re-read at vLLM `origin/main` `185cada36b`). CPU-only. No GPU
lease was taken and none was needed.

**What was fetched, exactly.** The complete
`model.safetensors.index.json` of `dots-studio/dots3-note-prev` at revision
`1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b` (3436982 bytes, sha256
`95a364b468a93ccad6adcb9c3aa110cb7a1411c2575c334c39022f9f84d456e1`), then the
safetensors HEADER of every one of the 133 files it names — two HTTP Range
requests each, 8 bytes for the header length and then the header JSON,
**4770592 header bytes in total and not one tensor byte**. The checkpoint is
576886825984 bytes and was never downloaded. The committed fixture is
`tests/vllm/models/fixtures/dots3_note_prev/index_full.json`; it collapses ONE
index, the routed-expert index, to `{E}` with the member count beside it, and
expands back to exactly 38006 names. Every backbone layer, vision block and
audio layer is a separate entry on purpose: W2 exists to MEASURE that the
layers repeat, and a fixture that collapsed them would assume the answer.

The released `config.json` was re-fetched at the same revision and is
byte-identical to the committed fixture, sha256
`99b7de680dd456111c36efb8749f8ae7177328e97b65a3e39a6700cbc1173833`.

**The gate, met.** `test_dots3_note_scaffold` — **26 cases / 110821
assertions**, CPU-only, no GPU, no checkpoint (19/3876 at W1). The accounting
reads **38006 / 38006**: 35381 language, 2195 vision, 430 audio, zero
unaccounted, zero missing, zero duplicated, zero invented. All three buckets are
asserted BY NUMBER in every case that touches them, and the whole 38006-name set
is driven through `ModelRegistry::Resolve(...).factory->load_weights` as well as
through the classifier, so the map is proved reachable and not merely correct.

**One thing changed shape rather than only growing.** W1 classified the towers
with two prefix literals and two integer counters. A counter cannot say whether
2625 weights are deferred on purpose or lost, so `Dots3NoteDeferredTowers()`
is now a table of records — prefix, the one file the tower ships in, the brick
that owes it, and what it is — and `AccountDots3NoteTensors` dispatches on that
table. The load refusal prints it, so an unknown tensor is distinguishable from
a deferred one in the message a user gets.

#### The three facts the slice could not reach

1. **The backbone has exactly FOUR distinct layer shapes, and no layer breaks
   the pattern.** Grouping all 47 layers by their (suffix, dtype, shape) set
   with the expert index collapsed gives `{0}` (dense MLP + full attention, 19
   tensors), the 12 full+MoE layers `{1, 5, 9, ... 45}` (789 each), the 33
   sliding+MoE layers (784 each) and `{46}` (18). W1 recorded "the remaining 42
   backbone layers repeat layers 1/2 exactly" as a claim about a checkpoint
   nobody had read. It holds. A fifth class would have meant the port was
   reading some layer with the wrong map.

2. **The full/sliding split derived from the WEIGHTS matches
   `config.layer_types` exactly.** The DSA indexer ships only on the full class,
   so the shipped `self_attn.indexer.wk.weight` names give the schedule
   independently of the config: `{0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41,
   45}`, 13 layers, and `q_b_proj` is [24576, 1024] on every one of them against
   [16384, 1024] everywhere else. Two independent released artifacts agreeing is
   a stronger statement than either alone, and §1.1's table is now measured
   rather than transcribed.

3. **The checkpoint carries 62 F32 tensors in TWO families, not one.** W1 saw
   one and predicted the family. The language tower's 45 are
   `mlp.gate.e_score_correction_bias`, one per MoE layer, layers 1 to 45 — as
   predicted. The other 17 are **`vision_encoder.blocks.{25..41}.mlp.router_bias`**,
   which W1's language-only slice could not see at all. Their widths are the
   pyramid's own routed-expert counts, 4, 8, 12 … 64, 64, which is §1.2's
   `pyramid_num_routed` confirmed from the weights. The vision MoE keeps its
   router statistics in fp32 inside an otherwise BF16 tower, and a loader that
   resolved one dtype for the checkpoint would read them wrong with no shape
   change and no error. Audio is BF16 throughout, and the census closes: 37944
   BF16 + 62 F32 = 38006, with no third dtype anywhere.

   **This is NOT R5, and an earlier draft of this section said it was.** R5 and
   §2.4 are about the FP32 **dynamic activation scales** inside
   `note_vision_fused_moe_fp8` — a quantized-path memory format. The bf16
   checkpoint carries no scale tensors at all; every one of its 38006 entries is
   a named parameter. `router_bias` is a learned fp32 parameter, which is a real
   and useful finding and a different one. **R5 stays entirely owed by W6**, and
   a W6 implementer must not read this row as confirming it.

#### And a fourth, which is a finding rather than a confirmation

**The released index declares an indexer RoPE layout that nothing reads.** Its
`metadata` block carries `"indexer_rope_layout": "leading"` and
`"indexer_rope_converted_from": "tail"` beside `total_size`. `git grep
indexer_rope_layout` over vLLM `origin/main` returns nothing, so upstream never
reads either key: it is the publisher stating how the DSA indexer's `wq_b` and
`wk` are laid out along the 128-wide index head, and saying the published
weights were re-ordered to get there.

It agrees with what upstream's code does anyway. `DeepseekV2Indexer` splits both
`q` and `k` as `[..., : rope_dim]` for the rotated half and `[..., rope_dim :]`
for the rest (`deepseek_v2.py:805`, `:814`, `rope_dim` = 64 of `index_head_dim`
= 128), which is a LEADING slice. **This is not §4 trap 2.** Trap 2 is about
which PAIRS the rope rotates, GPT-J against NeoX. This is about which HALF of
the head is rotated at all. Both are numerically silent, they are independent,
and §6.4 says this row has no oracle that could catch either.

W2 pins both values in an assertion so a re-published checkpoint cannot flip the
layout silently, and consumes neither, because W2 writes no maths.
[#1846](https://github.com/mudler/vllm.cpp/issues/1846) owns it and W3 owes the
slice.

#### What the two towers actually ship, for W6 and W7

W2 does not port either tower and writes no maths for them. It does read their
whole tensor list, because that is what "named deferral" has to mean, and four
of those facts are worth stating where the brick that owes them will look.

**Vision (2195 tensors, `model-vision.safetensors`).** 42 blocks, each with
`norm_1`, `norm_2`, `attn.{qkv, proj, q_norm, k_norm}` and an MLP. `qkv` is
[4608, 1536] — one fused projection, no bias — and `q_norm`/`k_norm` are [64],
so `use_qk_norm` acts per head at head_dim 64 over 24 heads. Blocks 0 to 24 are
DENSE with `mlp.{fc1, fc2, fc3}` at [4224, 1536] / [1536, 4224] / [4224, 1536],
a three-tensor SwiGLU rather than a gate/up/down triple. Blocks 25 to 41 are
MoE, with `mlp.experts.{E}.{fc1, fc2, fc3}` at the `moe_intermediate_size` of
2112 and a router that is `mlp.gate_weight` + `mlp.router_bias` — NOT the
`mlp.gate.weight` + `mlp.gate.e_score_correction_bias` spelling the language
tower uses. Outside the blocks: `patch_embed.proj` [1536, 3, 14, 14] with a
bias, `patch_embed.norm`, `post_trunk_norm`, and the patch-merger adapter
`adapter.{ln_q, mlp.0, mlp.2}` folding 4x1536 = 6144 to 5120.

**Audio (430 tensors, `model-audio.safetensors`).** 32 encoder layers of
`self_attn.{q_proj, k_proj, v_proj, out_proj}` at [1280, 1280], with a bias on
q, v and out and NONE on k — Whisper's own convention. `fc1` is [10240, 1280]
against `fc2` [1280, 5120], so the SwiGLU gate and up are packed into one
tensor at twice the 5120 `encoder_ffn_dim`. The stem is
`conv2d1` [480, 1, 3, 3], `conv2d2` and `conv2d3` [480, 480, 3, 3], then
`conv_out` [1280, 7680] = 16 x the 480 `downsample_hidden_size`. The adapter is
`audio_adapter.proj.{0, 1, 3}`, 1280 to 5120.

**There is NO learned positional embedding in the audio tower**, and that is
checkpoint and upstream agreeing rather than an absence to explain: at
`nvidia/audio_encoder.py:507-519` `DotsSpeechEncoder` sets
`self.embed_positions = None` when `use_rope` is true, and the released
`audio_config` sets it true. W7 must not go looking for one.

#### The mutation table

Every mutation was applied to the tracked source — or, for the `F` rows, to the
committed fixture — rebuilt, run, and reverted, with the tree verified
byte-for-byte afterwards. **The compiler exit status is printed beside each
row**, because a mutation that fails to build reads as a passing test and this
project has been bitten by that repeatedly. Every row compiled.
`cases`/`assertions` are what `doctest` reported failing.

| id | mutation | compiler exit | result | cases | assertions | first failing case |
|---|---|---:|---|---:|---:|---|
| M1 | the towers are counted as LANGUAGE | 0 | RED | 2 | 9 | W2: all 38006 tensors … are claimed |
| M2 | the audio tower is dropped from the deferral table | 0 | RED | 4 | 12 | W2: all 38006 tensors … are claimed |
| M3 | the vision deferral names the WRONG brick (W7) | 0 | RED | 2 | 2 | W2: the two tower files are NAMED W6/W7 deferrals |
| M4 | the vision deferral names the WRONG file | 0 | RED | 1 | 3 | W2: the two tower files are NAMED W6/W7 deferrals |
| M5 | the nextn layer is emitted with the FULL attention set | 0 | RED | 5 | 10 | enumeration: all 1614 tensors of the released slice |
| M6 | every backbone layer is treated as MoE (`first_k_dense_replace` ignored) | 0 | RED | 7 | 19 | config: the REAL released config.json parses |
| M7 | every backbone layer is treated as FULL attention | 0 | RED | 8 | 86 | config: the REAL released config.json parses |
| M8 | the headwise gate `g_proj` is dropped from the name map | 0 | RED | 6 | 16 | enumeration: all 1614 tensors of the released slice |
| M9 | `k_rope_only_layernorm` is dropped from the name map | 0 | RED | 6 | 15 | enumeration: all 1614 tensors of the released slice |
| M10 | the MoE shared expert is dropped from the name map | 0 | RED | 5 | 13 | enumeration: all 1614 tensors of the released slice |
| M11 | one routed expert per MoE layer is dropped (255, not 256) | 0 | RED | 5 | 13 | enumeration: all 1614 tensors of the released slice |
| M12b | a VISION tensor is added to the language name map | 0 | RED | 5 | 14 | enumeration: all 1614 tensors of the released slice |
| F1 | FIXTURE: one vision `router_bias` is re-typed BF16 | 0 | RED | 1 | 2 | W2: the memory format of the WHOLE checkpoint |
| F2 | FIXTURE: the indexer rope layout reads `tail` | 0 | RED | 1 | 1 | W2: the released index states an indexer RoPE layout |
| F3 | FIXTURE: one language tensor is moved into the vision tower file | 0 | RED | 1 | 1 | W2: the two tower files are NAMED W6/W7 deferrals |

**A sixteenth row came from the fresh review, not from W2.** R8 deleted the
PRODUCTION CALL SITE of `AccountDots3NoteTensors` in `LoadDots3NoteWeights` —
the one thing W2's own table never mutated, because W2 wrote the call — and it
came back RED. So the map is reached through the registry rather than only
exercised by helpers, and the reachability claim is measured rather than
asserted. The same review re-derived every number in this section against the
live release independently, including the 266 Range requests, the two fixture
hashes, all four bucket counts and the whole tower inventory below.

M1 is the row this brick exists for. It is the W1 review's M15 at full scale:
fold the towers into the language count and every "nothing was left over"
assertion stays green while 2625 weights go unloaded. It fires.

**One mutation came back GREEN, and the CODE changed rather than the table.**
W2 first wrote "a name cannot be both loaded and deferred" as a runtime
`VT_CHECK` inside `AccountDots3NoteTensors`. Deleting it left the whole gate
passing, because no config can make `EnumerateDots3NoteTensors` emit a
tower-prefixed name: every name it emits is `model.`- or `lm_head`-prefixed by
construction. That is production code no input reaches, which is the shape
AGENTS.md's reachability rule names, and the honest answer to a green mutation
is to remove what the gate cannot see rather than to keep it and note it. The
invariant is real, so the tower case asserts it over the real map, and **M12b
replaces the deleted row by injecting the defect the guard was meant to catch**
— a `vision_encoder.` name added to the map — which takes the gate red. The
guarantee is kept; the unreachable copy of it is not.

**M11 is a re-run, and the first attempt had a cause this spec got WRONG.** In
the batch it exited 135 with no parseable `doctest` summary. W2 wrote that up as
disk pressure — the box read 92% full at that moment — and **that was the wrong
cause**, corrected in place because `main` is never rewritten. The
[#1847](https://github.com/mudler/vllm.cpp/pull/1847) review found the real one
and reproduced it 3/3 **at 47 GB free and 61 GB RAM available**: `TempConfig`
and `TempCheckpoint` built their `/tmp` paths from a **per-process**
`static int counter`, so two concurrent runs of the same binary shared one
directory — both were watched sharing `/tmp/dots3_note_cfg_8`. Each constructor
rewrites a file the other has mmapped through `SafetensorsFile::Open` and each
destructor `remove_all()`s the other's, which is SIGBUS, exit 135, and a
block-buffered `doctest` summary lost with the process.

**That failure mode reads as NO RESULT, not as a failure**, which is why it is
worth more than the mutation row it corrupted: under §6.4 option B this row has
no oracle, this file is its only instrument, and a second agent building on the
same box is routine here. Both paths are now process-unique. The identical shape
in at least `test_laguna_nvfp4_loader`, `test_kimi_linear_scaffold`,
`test_loader_unaligned_offsets`, `test_ltx2_lora` and
`test_minimax_h3_video_fold` is [#1860](https://github.com/mudler/vllm.cpp/issues/1860),
not this row.

Re-run alone the same mutation compiled clean and took the gate red on 5 cases /
13 assertions, naming `model.layers.1.mlp.experts.255.down_proj.weight` and 135
unaccounted tensors. A crash is not a red test, so the row carries the reading
that has a summary behind it.

**The fix carries its own RED-before pair, measured here rather than inherited
from the review.** Two concurrent runs of the same binary, same box, same
minute:

| arm | compiler exit | run A | run B | `Status: SUCCESS` printed |
|---|---:|---|---|---|
| `UniqueTempDir` (fixed) | 0 | exit 0 | exit 0 | both |
| `static int counter` (RED) | 0 | exit 1 | **exit 135, `Bus error (core dumped)`** | **neither** |

Taken at **45 GB free and 34 GB of free RAM**, which settles the cause: the
first write-up blamed disk, and the crash reproduces with plenty of both. The
RED arm needed one extra edit to COMPILE — reverting the two call sites leaves
`UniqueTempDir` unused and `-Werror=unused-function` fails the build — and a
mutation that fails to build reads as a passing test, so `[[maybe_unused]]` was
added to the RED arm and its `compile_err=0` is recorded above beside the
result. Note what the RED row does NOT say: run A "failed" with exit 1 and run B
printed nothing at all. **Neither process printed a summary**, which is the
whole hazard — the mode this defect produces is *no result*, and no result reads
like a run that has not finished.

**Two more pieces of production code went the same way as M12, on the same
argument.** `Dots3NoteAccounting::deferred()` had no production caller — its
only three were in the test, two lines below assertions that already read
`acc.vision == 2195` and `acc.audio == 430` directly — and its second
`VT_CHECK` was unreachable by exactly M12's reasoning, so it is deleted rather
than staged (review F2). And the classifier's `else ++acc.audio` would have
counted a hypothetical THIRD registered tower as audio: the table decided
language-versus-deferred correctly and then inflated the wrong bucket. It now
dispatches on the table INDEX and reports a counter-less tower as UNACCOUNTED,
so the load refuses naming it instead of miscounting (review F3). That branch is
unreachable while the table has two entries, and it is written as a safe
degradation rather than as a guard this gate can prove.

**What the gate costs.** 26 cases, CPU-only, and at `-DCMAKE_BUILD_TYPE=Debug
-DCMAKE_CXX_FLAGS_DEBUG=-O0` it runs in about 170 s, of which roughly 135 s
predates W2. Three cases dominate and all three build a whole-tower synthetic
safetensors and drive it through the registry; the assertion count is not the
cost. W2 halved its own share by loading once and reading the message instead of
running a 38006-tensor load per `CHECK_THROWS_WITH_AS`, and by reporting one
assertion per defect class with the first offender named rather than one per
tensor — which also stops a single classifier defect printing 2197 lines.

### 4.5 W3 wrote the first maths on this row, against a reference and not a helper

**LANDED at W3** (`row/MODEL-MM-dots3-note-W3`,
`src/vllm/model_executor/models/dots3_note_attn.{h,cpp}`,
`tests/vllm/models/test_dots3_note_attn.cpp`, upstream read at vLLM
`origin/main` `06ecec7a84`). CPU-only. No GPU lease was taken and none was
needed: a reference-versus-implementation gate has no device in it.

**The upstream anchors are re-derived, and §2.2's are W0-era.**
`git log 185cada36b..06ecec7a84 -- vllm/models/dots3_note/` is EMPTY, so the
dots3 sources are byte-identical to what W2 read; the line numbers differ
because §2.2 was written against `9035151d6`/`170592a93`. At `06ecec7a84`:
`_forward_note_mla` is `model.py:135-201` (not `:135` with the gate at
`:246-262`), the two LoRA rescales are `:155` and `:159`, `k_rope_only_layernorm`
is applied at `:160` and built at `:299-301`, the headwise gate is `:190-197`,
and the `is_sparse` guard is `:171`. The indexer is `deepseek_v2.py:751-842`,
its `k_norm` is built at `:708`, the rope polarity is `:1159`, and the softmax
scale `qk_head_dim ** -0.5` is `:1026`. Spec R2 anticipated exactly this drift;
the anchors live in `dots3_note_attn.h` beside the code that uses them.

**What landed, and what did NOT.** `dots3_note_attn.{h,cpp}` is a portable HOST
reference of `_forward_note_mla`'s full-attention arm, in double throughout. It
is **not** on the decode path: `Dots3NoteModel::ForwardDevice` still refuses by
name, and the last case of the gate asserts that refusal so the boundary is
executable rather than a comment. The device wiring needs the padded sparse MLA
backend over a heterogeneous KV cache — W4's brick — and W4 also owes the
`mla::ForwardMlaAttentionBlock` extension, because three of the four deltas sit
INSIDE that seam: the two rescales and `k_rope_only_layernorm` land between its
projections and its RoPE, and the headwise gate between its attention and its
`o_proj`. Extending a SACRED-gated shared block for branches no device forward
would exercise buys untested optional paths and no gate; `deepseek_v4_dsa.{h,cpp}`
set that precedent and recorded it. Both items are under `## Owed`.

The indexer's SELECTION math is not a second copy of anything: it routes through
`deepseek_v4::DsaIndexerWeightFold` / `DsaIndexerLogits` / `DsaTopkSelect`,
which are ports of the same `layers/sparse_attn_indexer.py` and
`v1/attention/ops/triton_fp8_mqa_logits.py` dots3-note reaches through
`deepseek_v2.py::Indexer`. dots3-note's indexer delta is the rope GEOMETRY, not
the math.

**The gate, met.** `test_dots3_note_attn` — **12 cases / 198 assertions**,
CPU-only, no GPU, no checkpoint, no speed claim. The geometry is resolved from
the RELEASED `config.json` through `ModelRegistry::Resolve(...)`,
`factory->parse_config` and `ParseDots3NoteParams`, never typed by hand.

**The reference is independent, concretely.** It is transcribed from the python
listed above, and it is a different algorithm wherever a different one exists:
it rotates with a **complex multiply** and angles recomputed per element rather
than from a cos/sin cache, it softmaxes **without the max subtraction** in
`long double`, it selects the top-k by a **full stable sort** rather than a
partial selection, and it accumulates every dot product in `long double`. The
two arms agree to **1.7e-16 to 3.2e-16** relative across every traced
intermediate — `q_c`, `kv_c_normed`, `k_pe`, `q`, the attention output, the
gated output and the layer output — with the indexer's selection identical in
every one of the 24 slots. Under §6.4 option B that is still only two files
agreeing, so every mechanism ALSO carries a property a plausible-but-wrong port
breaks, which is a statement about the mechanism rather than about the files.

| Mechanism | Upstream | The property, measured |
|---|---|---|
| §4 trap 5, the two LoRA rescales | `model.py:155`, `:159`; scalars `:305-307` | dropping them moves the output 0.293 and `q_c` by exactly `(s-1)/s`; applying them BEFORE the norm instead is a **no-op** (7.3e-15), because RMSNorm is input-scale-invariant — so the gate distinguishes a MISSING multiply from a MISPLACED one |
| `k_rope_only_layernorm` | `model.py:160` | the whole layer is INVARIANT to a 7.5x rescale of the `kv_a_proj_with_mqa` rows that produce `k_pe` (5.0e-14). DeepSeek, which has no such norm, is NOT: the same rescale moves it 0.272, and dropping the norm moves ours 0.117. This property needs no reference at all |
| the headwise gate | `model.py:190-197` | `gated[t,h,d] / attn_out[t,h,d]` is constant over `d` and equals `sigmoid(logit[t,h])` to 5.6e-17. A lane-wise, transposed or wrongly-broadcast gate breaks it; broadcasting head 0 to every head moves the output 0.419 |
| §4 trap 2, `indexer_rope_interleave` | `deepseek_v2.py:1159` | GPT-J against NeoX changes **7 of 24** selection slots and the output by 0.754 |
| #1846, the LEADING rope slice | `deepseek_v2.py:804-805`, `:813-814` | LEADING `[0,64)` against TAIL changes **10 of 24** selection slots and the output by 0.793. It also differs from the NeoX answer, so the pairing and the slice are two independent questions and neither subsumes the other |
| `is_sparse` | `model.py:171` | dense causal attention is a different answer by 0.392 |

**The instrument says what it measured.** Of the 8 query rows, **5** really
prune (tokens 0-2 have 1, 2 and 3 causal candidates against `index_topk` 3, so
they take the short-context all-select path). Of those 5, **2** are decided by a
strict margin and **3** by an exact tie at zero — the indexer logit is
`sum_h w * ReLU(dot)`, so a key whose every head dots negative scores exactly
`0.0`. A tie at `0.0` is representable in float and double alike and both arms
break it by the smaller key index, so it cannot flip; the strict margins are
bounded at **1.29e-3**, three orders above the ~1e-7 where the implementation's
float-narrowed logits could matter. The largest raw attention score is 1.15, so
the reference's max-subtraction-free softmax is inside `exp`'s comfortable
range. All four numbers are printed by the gate, not assumed by it.

**Two instrument defects the RED arm found, and both are the same shape.** The
RMSNorm epsilon sits INSIDE the root (`ir/ops/layernorm.py:17-18`), so
`mean((s*x)^2) + eps` is not `s^2 * (mean(x^2) + eps)` and the scale invariance
is exact only as eps goes to zero. At the tiny bench's deliberately large
`rms_norm_eps` of 1e-3 the two properties built on that invariance measure
7.2e-4 and 4.7e-3 rather than zero. Both now run on a second bench whose only
difference is a negligible epsilon, and the epsilon-limited number is REPORTED
beside the clean one. The large epsilon stays on the main bench, because it is
what makes the epsilon's own placement mutable (M18).

#### The mutation table

Every mutation was applied to the tracked source, rebuilt, run, and reverted,
with the tree verified byte-for-byte afterwards. **The compiler exit status is
printed beside each row**, because a mutation that fails to build reads as a
passing test. Every row compiled. `cases`/`assertions` are what `doctest`
reported FAILING; the total assertion count varies between rows because a
`REQUIRE` aborts its case.

| id | mutation | compiler exit | result | cases | assertions | first failing case |
|---|---|---:|---|---:|---:|---|
| R0 | RED-FIRST: all four deltas dropped at once — plain DeepSeek MLA | 0 | RED | 4 | 12 | agrees with the independent reference |
| M1 | the q LoRA rescale is dropped | 0 | RED | 2 | 6 | agrees with the independent reference |
| M2 | the kv LoRA rescale is dropped | 0 | RED | 1 | 4 | agrees with the independent reference |
| M3 | both rescales move BEFORE their layernorm | 0 | RED | 2 | 9 | agrees with the independent reference |
| M4 | `k_rope_only_layernorm` is dropped | 0 | RED | 2 | 7 | agrees with the independent reference |
| M5 | `k_rope_only_layernorm` is applied AFTER the rope | 0 | RED | 1 | 4 | agrees with the independent reference |
| M6 | the headwise gate is dropped | 0 | RED | 2 | 4 | agrees with the independent reference |
| M7 | head 0's gate is reused for every head | 0 | RED | 2 | 4 | agrees with the independent reference |
| M8 | the gate reads a perturbed hidden state | 0 | RED | 1 | 2 | agrees with the independent reference |
| M9 | SILENT: the sigmoid becomes a hard step at 0.99/0.01 | 0 | RED | 2 | 3 | agrees with the independent reference |
| M10 | §4 trap 2: the indexer rope flips to NeoX | 0 | RED | 3 | 6 | the FULL geometry comes off the RELEASED config |
| M11 | #1846: the indexer rotates the TAIL slice | 0 | RED | 3 | 8 | the FULL geometry comes off the RELEASED config |
| M12 | the indexer's `k_norm` becomes an RMSNorm | 0 | RED | 1 | 4 | agrees with the independent reference |
| M13 | the indexer reads the UNRESCALED `q_c` | 0 | **GREEN** | 0 | 0 | — see below |
| M14 | the layer goes DENSE causal — the top-k stops being the mask | 0 | RED | 2 | 4 | agrees with the independent reference |
| M15 | the MLA rope rotates the LEADING lanes of the 192-wide head | 0 | RED | 1 | 4 | agrees with the independent reference |
| M16 | REACHABILITY: the dims stop reading the released params | 0 | RED | 2 | 4 | the FULL geometry comes off the RELEASED config |
| M17 | the softmax scale picks up a YaRN mscale² | 0 | RED | 2 | 4 | the FULL geometry comes off the RELEASED config |
| M18 | the RMSNorm epsilon moves OUTSIDE the root | 0 | RED | 2 | 8 | agrees with the independent reference |
| R8 | the indexer's `k_norm` epsilon moves 1e-6 -> 1e-3 | 0 | **GREEN before the F1 fix, RED after** | 1 | 1 | the FULL geometry comes off the RELEASED config |

**R0 is the RED-first arm and it ran BEFORE the green one.** With all four
deltas neutralised the gate reads 12 cases / 4 failed and 198 assertions / 12
failed, exit 1, compiler exit 0. That is a real run and not an inference.

**M9 is the row the reference earns its keep on.** A sigmoid replaced by a hard
step keeps the gate in `(0,1)`, keeps it per-head, and keeps
`gated/attn_out == sigmoid`-shaped, so every property assertion in the headwise
case still holds. Only the comparison against the reference sees it.

**M13 came back GREEN, and the CODE changed rather than the table.** Feeding the
indexer the UNRESCALED `q_c` — moving §4 trap 5 from before the indexer to after
it — changes nothing, and the reason is an invariance rather than a hole: the
logit is `sum_h w[t,h] * ReLU(dot(q[t,h,:], k[s,:]))`, so a POSITIVE rescale of
`q_c` multiplies every logit in a row by one constant and the argmax does not
move. The logits' only consumer is the top-k, so `q_lora_scale` reaches the
output through the MLA scores and through nothing else. A comment in
`dots3_note_attn.cpp` claimed the opposite and is corrected, and the guarantee
the mutation actually probed is now ASSERTED rather than written down: scaling
`indexer_wq_b` by 4.0 leaves the selection and the whole layer output
byte-identical, with each of the 36 finite logits scaled by EXACTLY 4 — the
factor is a power of two on purpose, so the ratio is an equality and not a
tolerance. The forward keeps mirroring upstream and passes the rescaled `q_c`;
the invariance means the mirror is unobservable here, which is worth stating and
is not a reason to diverge.

**M16 is the reachability row.** It deletes the production read of the released
params inside `Dots3NoteFullAttnDimsFrom`, so the layer's geometry stops coming
from `config.json`, and the gate goes red. The honest limit stays honest:
nothing in `ModelRegistry::Forward` reaches this code at all, which `## Owed`
records and the last gate case asserts.

**No regression on the sibling gate.** `test_dots3_note_scaffold` re-ran at this
head: 26 cases / 110818 assertions / 0 failed.

#### What the fresh review added, and the two LOW findings it closed

The review returned PASS and proved the reference's independence in the STRONG
direction rather than by reading it. **R9 mutated the SHARED helper the
implementation routes through and the reference does not** — it dropped the ReLU
inside `deepseek_v4::DsaIndexerLogits` — and the gate went RED. So the reference
VALIDATES the shared helper instead of agreeing with it, which is the inverse of
the shared-helper failure mode AGENTS.md warns about. **R7** is the other
decisive one: the hard-step sigmoid keeps the gate in `(0,1)`, keeps it
per-head, and keeps `gated/attn_out == trace.gate`, so every property assertion
still passes and ONLY the two reference comparisons fire — which is the
reference earning its keep, stated as a measurement rather than as a hope. The
review also re-derived every mechanism from upstream itself rather than trusting
the transcription, and confirmed M13's algebra: `ReLU(s*x) = s*ReLU(x)` for
`s > 0` preserves the ranking INCLUDING exact ties, whatever the sign of `w`.

It also found the not-extending-the-MLA-seam decision right for a **stronger**
reason than this spec gave. `mla::ForwardMlaAttentionBlock` is a DEVICE seam
over a paged cache with a decode backend, so a host, cache-free, `double`
reference cannot route through it at all, and it carries no sparse/top-k mask —
that is **four of four** deltas it cannot represent, not three. The `## Owed`
entry says three; the count is corrected here rather than in place, because the
three-of-four reading is what the code comments argue and a reader should see
both.

**F1 — LOW, closed in flow. `indexer_k_norm_eps` was the one shared scalar no
case pinned.** R8 moved it three orders of magnitude, 1e-6 to 1e-3; the code was
REACHED — the min strict selection margin shifted 1.29e-3 to 1.16e-3 — and the
gate stayed 12/198 green, because BOTH arms read the same wrong number. Two arms
drifting together is the shared-helper failure mode in a different hat, and
under §6.4 option B this gate is the only correctness instrument the row has.
Closed in both directions: the reference now carries the upstream LITERAL
(`deepseek_v2.py:708`, `LayerNorm(head_dim, eps=1e-6)`) instead of reading the
implementation's field, and the released-config case pins
`d.indexer_k_norm_eps == 1e-6`. Every other shared scalar was already pinned —
both LoRA scales, both rope polarities, `rope_theta`, `softmax_scale()`,
`IndexerRopeOffset` and `rms_norm_eps`.

**The two halves are NOT redundant, and re-running R8 says which one works.**
With both in place R8 goes red on **one** assertion — the released-config pin —
and the reference comparison stays GREEN. That is not a weakness in the
reference; it is the mechanism: the indexer's only product is the SELECTION, and
at this fixture the three-order eps shift moves the margin without moving the
chosen keys, so both arms produce identical output whatever they think the
epsilon is. So the PIN is what catches a wrong value today, and the LITERAL in
the reference is a drift guard: it stops a future edit from changing the
epsilon in one place and having both arms follow it. An earlier draft of this
brick's commit message claimed either edit alone would red the mutation; that
was wrong, and measuring it rather than asserting it is what found so.

**F2 — LOW, closed in flow, and pointed, because this brick's own headline is
that W0's anchors were stale.** Nine line citations in NEWLY WRITTEN comments
pointed at unrelated code at `06ecec7a84`. Every one was re-derived here rather
than adopted from the review, and that mattered twice: the reviewer offered
`deepseek_v2.py:1025` for the MLA softmax scale and this brick had written
`:1027`; `:1025` is blank, `:1027` is `self.max_position_embeddings`, and the
line is **`:1026`** — two readers miscounted the same line in opposite
directions. And an earlier draft of the §2.2 correction below cleared items 1
and 2 as "verified unchanged" without checking them; they were stale too.

| claim | was | is, at `06ecec7a84` |
|---|---|---|
| MLA softmax scale `qk_head_dim ** -0.5` | `deepseek_v2.py:1027` | **`:1026`** (inside `DeepseekV2MLAAttention`, `:982-1229`) |
| `Indexer.forward` | `deepseek_v2.py:788-828` | **`:751-842`** |
| the indexer's concat back to `[q_pe, q_nope]` | `deepseek_v2.py:818-819` | **`:825`** |
| the sliding class sets `is_sparse` False | `model.py:430-432` | **`:432-434`** |
| the top-k selector | `sparse_attn_indexer.py:488-497` (an XPU branch) | **`:509-518`** (`ops.top_k_per_row_prefill`) |
| the indexer weight fold | `sparse_attn_indexer.py:203-207` | **`:203-206`** |
| ReLU then head-sum | `triton_fp8_mqa_logits.py:125-132` | **`:129-132`** |
| `RotaryEmbedding.forward_static` | `base.py:178-201` | **`:161-201`** |
| the MLA rope construction | `deepseek_v2.py:1104-1110` | **`:1104-1109`** |

Verified correct as written and left alone: `model.py:135`, `:155`, `:159`,
`:160`, `:171`, `:190-197`, `:198-200`, `:201`, `:219`, `:230-238`, `:292-298`,
`:299-301`, `:303-307`; `deepseek_v2.py:708`, `:804-805`, `:813-814`,
`:1155-1159`; `ir/ops/layernorm.py:10-21` and `:17-18`; `base.py:80-103`;
`common.py:169-181`.

**§2.2's own four citations were ALL stale and are corrected in place** — see
the note under that list. `deepseek_v4_dsa.h:40` carries the same
`sparse_attn_indexer.py:488-497` rot and is NOT touched here: it is another
row's file and fixing it in this PR would be scope creep. No checker can catch
any of this — `check-symbol-anchors.py` verifies that a named symbol still
exists and says in its own docstring that it does not verify line citations,
which is [#1139](https://github.com/mudler/vllm.cpp/issues/1139).

## 5. Gates

**Correctness first, and the gate form is chosen by measurement, not in advance**
— capture K=5 greedy oracle runs; deterministic ⇒ STRICT token-exact, otherwise
the ratified near-tie distributional bar. That decision cannot be made until §6
produces a running oracle.

Until then, each brick gates against an **independent in-test double-precision
reference** written from the upstream source, with RED-first mutation proof.
That is a consistency gate, not a correctness gate, and the spec says so at every
brick: a shared-helper comparison proves the two arms agree, never that either is
right.

Owed measurements when hardware exists: token gate; decode/prefill throughput and
TTFT against production-configured vLLM (never `--enforce-eager`); memory; and
the quantized arms including GGUF k-quants.

---

## 6. Oracles and hardware — the blocking section

### 6.1 The oracle is vLLM, past the pin

vLLM is the primary oracle and implements this model, so no secondary oracle is
admissible for it. But our parity pin is `555967922` (0.26.0.dev0, 2026-07-26,
[upstream-sync.md](../upstream-sync.md)), and the checkout at that SHA contains
only `dots_ocr.py` — verified, not assumed. dots3 exists on `main` only.

This row is therefore **beyond-pin**, alongside `KimiK3ForConditionalGeneration`
and `MuseGlimmerForConditionalGeneration`. Gating it requires a pin advance,
which is a full sync cycle with every affected row reconciled — not a version
transcription. Worse for scheduling: upstream is still changing this code
(`#52172` landed 2026-08-13, the day before this spec), so a pin advance taken
now captures a moving implementation.

### 6.2 Memory — the model does not fit anywhere we own

| Vehicle | Size | Fits GB10 (~119 GiB)? | Fits Thor (~122 GiB)? |
|---|---|---|---|
| `dots3-note-prev` bf16 | ~576 GB | no | no |
| `dots3-note-prev-fp8` | ~290 GB | no | no |
| hypothetical ~2 bpw GGUF (ours) | ~75–90 GiB + towers | plausible | plausible |

Upstream's own recipe is `--tensor-parallel-size 8` on H100s. Two of our boxes
together are ~240 GiB and there is no TP-over-LAN path here, so aggregating them
is not a plan.

**The consequence, stated plainly: there is no configuration in which the vLLM
oracle runs this model on hardware this project has.** Our own arm may well run
at ≤2 bpw — that is the ds4flash IQ2_XXS pattern — but an arm with no oracle
beside it produces no parity number. This is the same wall
[deepseek-v4-flash.md](deepseek-v4-flash.md) hit at 156.7 GiB, three times worse,
and unlike that row there is no smaller published checkpoint to retreat to.

### 6.3 Thor as the designated e2e host — what it can carry

Developer directive 2026-08-14: use Thor as the CUDA host for end-to-end
verification. Reach it as the fleet device `thor:gpu0` through an `rc` lease and
never by `ssh`; [environment.md](../environment.md) carries the recipe. Probed
read-only over `ssh` on 2026-08-14, before that rule existed:

```
hostname   kairos-4db2      aarch64, 14 cores
memory     122 GB total, 118 GB available
disk       /home 918G, 123 GiB free   (/ is a 4.4G loop, 1.3G free)
toolchain  python3 present; NO nvcc, NO cmake, NO ninja, NO ~/venvs
nvidia-smi refuses under non-interactive ssh:
           "NvRmMemInitNvmap failed: error Permission denied"
```

**Two of those five lines are false now, and the block stays as the probe it
was rather than being corrected in place.** Free disk read 362 GB when it was
measured again on 2026-08-15, and `nvidia-smi` does not refuse. The points below
carry both corrections.

Three facts follow, and they are recorded rather than worked around.

1. **Thor cannot host the oracle for this model, and the reason is RAM.** The
   290 GB FP8 checkpoint is more than twice the box's 122 GB of memory, so the
   model cannot be resident whatever the disk holds. **The disk half of this
   argument is WITHDRAWN.** W0 wrote it as "290 GB exceeds both its 122 GB of
   RAM and its 123 GiB of free disk — the checkpoint will not even land", and
   the very next measurement of the same volume read 362 GB free, which is more
   than 290. On that number the checkpoint lands and then fails to load. Both
   probes were correct when they were taken, which is the point: free space is
   not a stable premise and a memory ceiling is. Designating the host does not
   change §6.2; it fixes *where our arm and our unit gates run*, which is a real
   and separate thing.
2. **Thor needed provisioning first** (W0.5) — **DONE 2026-08-15, and its
   recipe was REPLACED on 2026-08-19**; both the recipe and the `ctest` baseline
   live in [environment.md](../environment.md). Two corrections to the read-only
   probe above. `nvidia-smi` was never broken: it runs plainly inside a leased
   worker, exit 0, with no `NvRm` line at all, so `NvRmMemInitNvmap failed:
   Permission denied` was a privilege artefact of the unprivileged `ssh` shell
   and not a driver fault. And the toolchain does not go on the host at all —
   `/` is a read-only loop on an immutable image, so the box keeps no host CUDA
   and the CUDA 13.0.88 toolkit comes from the leased worker's own
   `/usr/local/cuda`. Free disk on `/home` measured 362 GB, not the 123 GiB the
   earlier probe read. Still owed if oracle work is ever wanted here: a vLLM
   build, which §6.2 says cannot serve THIS model regardless.
3. **Thor's standing traps apply.** `vm.overcommit_memory=1`: the kernel grants
   memory it cannot back and touching those pages takes the whole machine down
   (observed three times on 2026-08-11). Any run here is sized conservatively
   and never `-j` parallel across model gates. **The "zero swap" half of that
   trap is now stale** — a leased worker read 30 GiB of swap, all free, on
   2026-08-19 — but the reboots were observed and the hazard stands; see
   [environment.md](../environment.md) for the measurement and what it does and
   does not change.

What Thor *is* good for on this row: sm_110 runtime coverage (it is our only
non-GB10 CUDA host), our own low-bit arm end to end, and every unit/brick gate in
§7 — none of which need the 290 GB checkpoint.

### 6.4 How this row proceeds — DECIDED

**Developer decision, 2026-08-15: option B.** The row ports brick by brick
against independent references, ships our own low-bit arm on Thor, and carries
the end-to-end parity gate as an OPEN GAP rather than a satisfied one. A and C
below are recorded because the reasoning stays useful if the hardware position
changes.

What choosing B commits this row to, stated plainly so no later reader has to
infer it:

- **No performance number for dots3-note is claimable, on any axis, for as long
  as B holds.** Not a ratio, not a floor, not "comparable to". The oracle cannot
  run here (§6.2), so there is no denominator. A number measured against our own
  arm alone is a self-comparison and says nothing about parity.
- **Correctness claims are bounded by their instrument.** Each brick gates
  against an independent in-test double-precision reference written from the
  upstream source. That establishes that two implementations agree; it does not
  establish that either matches vLLM. Every brick says so in its own evidence
  cell rather than relying on this paragraph.
- **The e2e gate is owed, not waived.** It stays an open gap on the row until
  either the pin advances onto hardware that runs the model, or a smaller
  checkpoint appears. `## Owed` in this spec is where it lives.
- **No ceiling is declared** — unchanged, and the reason B is acceptable at all.

The alternatives, kept for the record:

- **A — rent 8×H100.** The only path to a real parity gate. Cost and data-egress
  are the developer's call; nothing in-repo authorizes it.
- **B — unit-gated bricks, e2e owed.** Port each brick against an independent
  reference, ship our low-bit arm on Thor, and record the parity gate as an open
  gap on the row. Honest, and it is what the row will do by default.
- **C — park at W0.** The row stays `SPIKE`, the scope is on record, and the work
  waits for either a smaller checkpoint or hardware.

Whichever is chosen, **no ceiling is declared** and the gap stays open.

---

## 7. Phases

W0 is this document. §6.4 is answered — option B — so the phases below are
dispatchable in order, under the constraints that answer imposes.

- **W0 — scope (this).** Arch map, reuse-vs-new, config traps, quant/HW fit,
  oracle plan, rows. **DONE.**
- **W0.5 — provision Thor. DONE 2026-08-15; recipe REPLACED 2026-08-19.** The
  recipe is [environment.md](../environment.md), under the Jetson Thor profile:
  one `rc run -d thor:gpu0` job that builds and tests inside the leased worker,
  which carries cmake 3.28.3, ninja 1.11.1, gcc 13.3.0 and python3, and which
  apt-installs the CUDA toolkit as a step because **the toolkit is NOT in the
  worker image** — both of this lane's jobs found nvcc already present and both
  were reading another job's leftover install in a long-lived container. **The
  recipe this row first landed is withdrawn, not merely superseded.** It
  prescribed `ssh` to the box plus `sudo -n docker build` and `sudo -n docker
  run` against a digest-pinned image, which bypasses the GPU lease and makes the
  fleet report the box free while a job is on it. The image was also
  unnecessary: a job installs the toolkit itself in one step, which is what
  `dgx:gpu0` jobs already do (#1213). A HOST toolchain was
  and remains rejected on evidence rather than taste — `/` is a read-only 4.4 G
  loop on an immutable Kairos image, so `apt install` into it does not exist.
  The build is CUDA-real, and **all three checks now run**: configure prints
  `CUDA target architectures: 110`, `libvllm.so` links `libcudart.so.13` and
  `libcublasLt.so.13` over 33 `*.cu.o` objects, and `cuobjdump --list-elf` over
  those objects reads **33 `sm_110` cubins across 33 objects scanned**. The
  third check could not be run as W0.5 wrote it, because `cuobjdump` was absent
  from the leased worker and yielded an empty result that looks like a clean
  one; installing `cuda-cuobjdump-13-0` explicitly, asserting the binary, and
  printing the object COUNT beside the histogram is what fixed it. That retires
  "30 objects, one `sm_110` cubin each" as an unverified 2026-08-15 claim — the
  shape was right, the count was stale. A kernel does run on the device:
  `test_cuda_backend` reports `sm_110`, `integrated=1`, `UnifiedMemory=true`,
  `DeviceMemoryIsHostAddressable=false`, 7/7 cases and 26/26 assertions.

  **The gate as written — "the existing suite passes there" — is NOT met, and
  it was the wrong gate.** Re-measured at `6756f9131` on 2026-08-23 inside an
  `rc run -d thor:gpu0` lease, the baseline is 598 tests, **573 passed /
  3 skipped / 22 red** (`ctest -j1 --timeout 1800 --output-on-failure`,
  632.35 s, job `8bf39567-9334-4f7e-aa27-43a2aa867bb7`, artifacts under
  `/mnt/nas_share/rc/thor-w05-955/out/`). It read 553 tests / 16 red at
  `0764ded2b` and 485 tests / 15 red at `2daa3287f`. **The staleness debt of
  [#955](https://github.com/mudler/vllm.cpp/issues/955) is PAID**; the table,
  the diff and the artifact paths live in
  [environment.md](../environment.md).

  **The 22 split into six causes, and the split is non-overlapping so it sums
  to 22.** Two entries belong to two descriptions at once and are counted ONCE,
  under GB10, with the second fact noted rather than added: `test_capi`, which
  is red on GB10 *and* improved `SEGFAULT` → `Failed` here, and `test_cuda_ops`,
  which is long-standing on GB10 *and* new to this host. **The referent of this
  rule moved at the 2026-08-23 measurement** — it used to be the three
  `qwen3_5_gdn_spec_routing` tests, whose double fact was their own
  `SEGFAULT` → `Failed` improvement at `0764ded2b`; this run they are simply
  unchanged, so they now sit under GB10 with nothing to double-count. An earlier
  draft's tally reached only 15 by counting such an entry under neither
  description cleanly.

  | Cause | Count | Tests |
  |---|---:|---|
  | no vendored FA-2 — the build correctly refusing what the arch lacks | 4 | `test_deepseek_v2_forward`, `test_ops_mla_prefill`, `test_ops_mla_chunked_context`, `test_mla_attention_block` |
  | the TEST hardcodes GB10 | 2 | `test_platform` (sm_12x family, and now also `supports_fa2_attention()`), `test_op_parity` (a dgx-only golden that runs anyway) |
  | already red on GB10, so not an sm_110 fact ([#907](https://github.com/mudler/vllm.cpp/issues/907)) | 6 | `test_linear_method`, `test_capi`, `test_cuda_ops`, and the three `qwen3_5_gdn_spec_routing` tests. `test_capi` ALSO improved `SEGFAULT` → `Failed`, and `test_cuda_ops` is new HERE while long-standing on GB10; both counted once |
  | the FP8 ops on an arch outside `VT_CUTLASS_FP8_ARCHS` ([#1725](https://github.com/mudler/vllm.cpp/issues/1725) — **not** [#960](https://github.com/mudler/vllm.cpp/issues/960), closed 2026-08-16 by `d607fec4c`, three days before the 2026-08-19 measurement) | 2 | `test_ops_fp8_cutlass`, `test_ops_matmul_fp8_block_cuda`. **They no longer CRASH**: `cffe59b02` made the portable tier ineligible on a backend whose device memory is not host-addressable, which is Thor. The residue is that the block-scaled op refuses generically rather than by name |
  | the live Marlin NVFP4 disagreement ([#962](https://github.com/mudler/vllm.cpp/issues/962)) | 1 | `test_ops_moe_grouped`, reproduced byte-identically at `bitdiff=15/32768` |
  | UNATTRIBUTED, now owned by [#1802](https://github.com/mudler/vllm.cpp/issues/1802) | 7 | `test_gguf_device_fit_reach` (since 2026-08-15), `test_serve_low_tools` (whose CAUSE changed — no longer the absent `shellcheck` of [#961](https://github.com/mudler/vllm.cpp/issues/961), which `73ada0df8` fixed), and the five that arrived at this measurement: `test_backend_cross_device`, `test_llama_embedding_fold`, `test_mtp_depth`, `test_qwen3_dflash2_draft`, `test_ops_attention_dense_fa2` |
  | **total** | **22** | |

  **Six names arrived, none departed, and three modes IMPROVED — every
  `SEGFAULT` on this box is gone.** That is the differential gate paying for
  itself in the direction it was designed to see: a count of names reads
  16 → 22 and scores six regressions, while the pairs read six arrivals, zero
  departures and three improvements.

  None of those can be
  made green by this row and none is this row's debt. Asking for "all green" on
  a host whose arch legitimately lacks features would either block every brick
  forever or invite someone to weaken a test to pass. **The gate that actually
  binds is therefore differential, over `(name, failure mode)` PAIRS: a row
  regresses on Thor when it adds a name, or changes a recorded mode FOR THE
  WORSE — worst first, `SEGFAULT`/`Subprocess aborted`/`Timeout`, then `Failed`,
  then `Skipped`/`Not Run`, then passing. A crash becoming a clean assertion
  failure is progress rather than a regression, and this lane's own run saw
  three such improvements, so the direction is not a hypothetical refinement.
  `Skipped` ranks BELOW `Failed`: a red test that starts skipping has stopped
  being measured, not started passing, and three tests already skip here for an
  absent checkpoint. A name leaving the list therefore counts as an improvement
  only when it has been SEEN to pass.** The list
  itself, with a mode and a first-failing assertion per test, is
  [#955](https://github.com/mudler/vllm.cpp/issues/955), the sm_110 counterpart
  of [#907](https://github.com/mudler/vllm.cpp/issues/907).

  **"Only if it lengthens the list" is what this gate said when W0.5 landed, and
  it is provably too weak.** Counting names scores a crash as no change. Between
  `5a0ffe9e3` and `2daa3287f` five tests went `Failed` → `SEGFAULT` with no name
  change, and the list grew by one only because the same upstream change also
  shipped a new test file. A name-counting gate would have scored
  [#960](https://github.com/mudler/vllm.cpp/issues/960) GREEN on five of the six
  crashes it introduced. The mode column is part of the baseline, not
  decoration, and it costs no extra measurement because `ctest` prints the mode
  beside every failure.

  That gate has to be re-derived, not remembered, and it has now moved three
  times. Two SHAs a few hours apart on 2026-08-15 read 484/14 → 485/15, because
  a change on `main` turned a clean FP8 refusal into a segfault. Four days later
  at `0764ded2b` it reads 553/16: three names arrived, two left, and the three
  `qwen3_5_gdn_spec_routing` tests went `SEGFAULT` → `Failed`, which a
  name-counting gate would have reported as a single regression and nothing
  else. Four days after that, at `6756f9131`, it reads 598/22 across 176
  commits: six names arrived, none left, and the last three crashes on the box
  became clean assertion failures. **Re-measure whenever the base moves across
  `src/`, `include/`, `tests/` or `CMakeLists.txt`.**

  Two consequences this row carries forward, and the second one has changed.
  Thor's MLA prefill throws rather than computes, so the W3/W4 attention bricks
  cannot be verified end to end here on the FA-2 path at all — their gate stays
  the in-test double-precision reference of §5, exactly as §6.4 requires under
  option B. And W9's blockwise-FP8 arm still has no native kernel on this box —
  but **the fallback no longer crashes.** At `6756f9131` the op refuses with
  `the portable CPU reference tier is NOT eligible: this backend does not report
  its device memory host-addressable`, which is the loud refusal the seam is
  supposed to produce. That arm remains owed rather than pending, for want of a
  kernel and not for want of a safe failure.
- **W1 — config + registry. DONE** (`row/MODEL-MM-dots3-note-W1`,
  [#699](https://github.com/mudler/vllm.cpp/issues/699)). `dots3_note_registry.cpp`
  is an additive TU with ONE `REGISTER_VLLM_MODEL` line and no edit to any shared
  array; `dots3_note.{h,cpp}` carry `Dots3NoteParams`, the name map and the
  refusing forward. All six §4 traps are gated RED-first — the evidence table is
  §4.1, and it also records what W1 measured that the spec did not know
  (§1.4 resolved, the nextn block's geometry, the one F32 tensor). §4.2 records
  why the config parsing did NOT go into `hf_config.cpp` as §3.2 item 6 proposed.

  **`Dots3NoteMTPModel` is deliberately NOT registered** and stays `INVENTORIED`
  on its own row: registering a speculator that cannot propose makes the engine
  accept a speculative config it then dies on mid-run. W10 owns it.

  **Gate, met.** `test_dots3_note_scaffold` — 15 cases, 3694 assertions, CPU-only,
  no GPU, no checkpoint. Registry resolve; the REAL released `config.json` parsed
  as a committed byte-for-byte fixture; six refuse-by-name cases for
  unrepresentable configs; the padded 1088-wide MLA row; and the tensor
  accounting, **1614/1614 claimed both ways** over a committed headers-only slice
  of the released index at revision `1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b`.

  **The slice is one layer of every class, which is MORE than the phase asked
  for and is said plainly rather than counted as the whole checkpoint:** the 4
  root tensors, layer 0 (dense MLP + full attention), layer 1 (MoE + full),
  layer 2 (MoE + sliding) and the nextn layer 46, out of the 38006 the
  checkpoint ships. Accounting a single class would have left the sliding
  geometry and the nextn tail unproven. The remaining 42 backbone layers repeat
  layers 1/2 exactly; the 2625 vision/audio tower tensors are **named deferrals**
  to W6/W7 in the loader's classifier, never silent drops. **W2 still owes the
  whole-index pass** and the two tower files.

  **The forward refusal is driven through the REAL model the factory returns.**
  `LoadDots3NoteWeights` runs the accounting and returns an UNMATERIALIZED model;
  `Dots3NoteModel::ForwardDevice` then refuses by name. Reaching the refusal by
  handing the entry point a fabricated `LoadedModel` subclass is undefined
  behaviour the moment the handle is opened — the defect UBSan caught on the
  NemotronH row ([#730](https://github.com/mudler/vllm.cpp/issues/730),
  [#775](https://github.com/mudler/vllm.cpp/issues/775)) — so a separate case
  uses a foreign model for the one guarantee that needs it: that the checked
  `ModelAs` downcast refuses a type mismatch by name.

  **NOT done at W1, and owed:** the row stays `SPIKE`. Advancing it to `ACTIVE`
  owes the §8.1 heading restructure, and that belongs to the brick where the
  forward stops refusing (W3+), not to a brick that only makes the arch resolve —
  the same reasoning `MODEL-MM-muse-glimmer-*` records for staying `SPIKE` with a
  whole text forward landed. `docs/USAGE.md` owes the checkpoint table when a
  capability becomes reachable; `docs/FEATURES.md` carries the arch row now.
- **W2 — weight map. DONE** (`row/MODEL-MM-dots3-note-W2`,
  [#699](https://github.com/mudler/vllm.cpp/issues/699)).
  `model.safetensors.index.json` read for real, in full: the 42 backbone layers
  W1's committed slice does not cover, and the two tower files
  (`model-vision.safetensors`, `model-audio.safetensors`) that W1 classifies as
  named W6/W7 deferrals. **Gate met: 38006/38006, buckets 35381 language /
  2195 vision / 430 audio, zero unaccounted, zero invented, zero duplicated.**
  §4.4 carries the evidence, the mutation table and the three things the slice
  could not see. §1.2's vision pyramid and §1.4 are now checkpoint-measured
  rather than config-inferred.
- **W3 — full-attention layer. DONE** (`row/MODEL-MM-dots3-note-W3`,
  [#699](https://github.com/mudler/vllm.cpp/issues/699),
  [#1846](https://github.com/mudler/vllm.cpp/issues/1846)).
  `_forward_note_mla`'s full arm as a portable HOST reference in
  `dots3_note_attn.{h,cpp}`: the two lora rescales, `k_rope_only_layernorm`, the
  headwise gate and the DSA indexer at `indexer_rope_interleave=True`. Gate met:
  `test_dots3_note_attn`, **12 cases / 198 assertions**, against an independent
  double-precision reference transcribed from the python, RED-first, 19
  mutations with the compiler exit status beside each. #1846 is discharged —
  `IndexerRopeOffset` is the consumer of the released index's
  `indexer_rope_layout: "leading"`, and the gate shows the tail slice picks
  different keys. §4.5 carries the evidence, the mutation table and the one
  mutation that came back green.

  **NOT reached, and named rather than implied:** the layer is not on the decode
  path and `Dots3NoteModel::ForwardDevice` still refuses by name. See `## Owed`.
- **W4 — sliding-window MLA.** The §2.3 stack: windowed metadata, the gather, the
  score mask, and the padded/heterogeneous KV spec. The largest brick; likely
  splits further. **W3 handed it two things**: the device wiring of the full arm
  (the padded sparse MLA backend is W4's, and the full layers cannot reach the
  decode path without it) and the `mla::ForwardMlaAttentionBlock` extension
  three of W3's four deltas need, because they sit inside that seam. Both are in
  `## Owed`.
- **W5 — MoE.** Ungrouped `noaux_tc` at 256/8 + the shared expert. Mostly
  routing our existing path at new dims.
- **W6 — vision tower.** Dense ViT half first, then the pyramid MoE and the
  FP32-scale FP8 formula. Reuses `qwen3_vl_vision` structure.
- **W7 — audio tower.** The `dots` stem deltas over our Whisper encoder.
- **W8 — MM front end + ABI.** Processor, video sampling, placeholder expansion,
  `<|audio_comp_*|>`, `include/vllm.h` surface, the example server as a thin
  client.
- **W9 — quantized arms.** Blockwise FP8 and the owed GGUF k-quant arm +
  converter.
- **W10 — MTP.** `Dots3NoteMTPModel` over the existing speculator seam.
- **W11 — gates.** Whatever §6.4 permits: full SACRED if A, the recorded-gap
  form if B.

---

## 8. Risks

- **R1 — the oracle never becomes reachable.** Primary risk; §6.4 is its only
  mitigation. Everything else is downstream of it.
- **R2 — upstream is still moving.** `#52172` landed the day before this spec.
  Pin against a specific `main` SHA per brick and record it, or the "ported from"
  anchors rot.
- **R3 — the four silent config traps** (§4). Mitigation: RED-first assertion per
  trap, before the consuming layer exists.
- **R4 — windowed MLA is genuinely new** and upstream needed two Triton kernels
  plus its own metadata builder to get it. Do not scope W4 as an increment on our
  paged full-attention MLA.
- **R5 — the vision MoE's FP32 activation scales** (§2.4) are the exact shape of
  a too-wide/too-narrow dtype defect that a token gate cannot see. Check the
  memory format against upstream explicitly, per
  [porting.md](../porting.md).
- **R6 — no llama.cpp comparison** for the GGUF arm, so the quantized floor has
  no external reference. Record it as a gap rather than substituting a different
  model's number.
- **R7 — Thor's overcommit + zero swap** takes the box down on an oversized run
  (§6.3). Size every run; never stack them.

## 8.1 A record constraint discovered while landing W0

Advancing this row to `ACTIVE` requires this spec to grow the **structured
headings** `check-agent-record.py` enforces for an active row: Scope, Upstream
chain, Our baseline, Port map, Tests to port, Gates, Dependencies. The W0 shape
above deliberately does not have them, and the checker accepts that at `SPIKE`.
Measured by mutating the row's state to `ACTIVE` in a scratch copy and reading
the seven errors it produced. Whoever takes W1 owes that restructure in the same
change as the lifecycle move, not afterwards.

## Owed

Carried openly under option B (§6.4), not waived:

- **The end-to-end parity gate against vLLM** — token-exact or the ratified
  near-tie form, chosen by measurement. Blocked on §6.2 (no host we own runs the
  oracle at any published precision) and on the beyond-pin position of §6.1.
  Owner: this row. Issue [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **Every throughput, latency and memory axis.** Open by construction while the
  gate above is owed; see §6.4 for why no number is claimable meanwhile.
- **The full-attention layer is NOT on the decode path.** W3 landed
  `_forward_note_mla`'s full arm as a portable host reference
  (`src/vllm/model_executor/models/dots3_note_attn.{h,cpp}`) with its gate;
  `Dots3NoteModel::ForwardDevice` still refuses by name and nothing in
  `ModelRegistry::Forward` reaches the new code. The wiring needs the padded
  sparse MLA backend over a heterogeneous KV cache, which is **W4**, and W4 also
  owes the `mla::ForwardMlaAttentionBlock` extension the three non-indexer
  deltas need (two `double` scales, one optional norm weight, one optional gate
  weight on `MlaBlockDims`/`MlaBlockWeights`). Owner: row
  `MODEL-MM-dots3-note-dots3-note-for-causal-lm`. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **The indexer's fp8 quantization, which neither arm can see.** Upstream
  quantizes the indexer's `q` per 128-element group to fp8 and folds the
  resulting `q_scale` back into `weights` before the logits are formed
  (`deepseek_v2.py:831-836`, `:840` — re-derived at `06ecec7a84`). Both W3 arms
  are unquantized, so a selection flip caused by fp8 rounding is invisible to
  BOTH of them, which is the same class as the bf16 debt below rather than a
  gap in the gate. Note the interaction with §4.5's M13: the argmax is invariant
  to a POSITIVE rescale of `q_c`, so `q_lora_scale` cannot move the selection —
  but fp8 rounding is not a rescale, and that invariance does not extend to it.
  Owner: this row. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **The bf16 memory format of the four deltas.** The W3 reference is double
  throughout, which is strictly wider than the model path. Upstream computes the
  headwise gate's sigmoid in **fp32** and casts back to the activation dtype
  (`model.py:196`), and both LoRA rescales multiply a bf16 tensor by a python
  float, i.e. **in bf16** (`model.py:155`, `:159`). A double reference cannot
  see either, and `porting.md` says a token gate cannot see a too-wide dtype at
  all. The device brick owes the exact widths. Owner: this row, W4. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).

## 9. Stop conditions

- Any brick whose only available comparison is a shared helper stops and says so
  rather than claiming correctness.
- If W2's tensor accounting disagrees with §1 geometry, stop and reconcile this
  spec before writing a layer.
- If a pin advance is attempted and any affected row cannot be reconciled, the
  advance stops; this row waits rather than moving the pin around it.
- No throughput number is quoted for this model until a same-tool trace exists
  for both arms on identical workloads.

## Now

W0 complete; **§6.4 answered on 2026-08-15 with option B**, so the row is no
longer blocked on a decision. **W0.5 landed the same day.** **W1 has since
landed code** — see the `W1 — DONE` paragraph at the end of this section for
what that did and did not change. The row still reads `SPIKE`, and the reason
moved: it is no longer "W1 has not landed", it is that making an architecture
resolve is not porting a model.

**W0.5 — DONE, and its RECIPE was replaced on 2026-08-19.** `thor:gpu0` builds
vllm.cpp with CUDA ON for sm_110, runs kernels on the device, and has a recorded
`ctest` baseline. The full recipe — how to stage a tree for the lease, the one
`PATH` prepend that produces `nvcc`, the build and `ctest` flags, the corrected
`nvidia-smi` reading and the per-test failure table — is in
[environment.md](../environment.md) under the Jetson Thor profile, not here:
Thor is the project's only non-GB10 CUDA host, so it belongs to every row that
wants sm_110 coverage rather than to this one.

**What the first version of W0.5 got wrong, corrected in place because `main`
is never rewritten.** It prescribed `ssh` to the box plus `sudo -n docker build`
and `sudo -n docker run` against a digest-pinned image. That reaches a fleet
device outside its lease, so the fleet reports `thor:gpu0` free while a job is
on it, and the image was never needed — the leased worker already carries nvcc
13.0.88, cmake 3.28.3, ninja 1.11.1 and python 3.12.3. The recipe is deleted
rather than kept as an alternative. Two of its factual claims went with it: that
`nvidia-smi` "dies" unprivileged, which was reading a successful report's stderr
as a verdict, and that `shellcheck` was in the recorded Dockerfile, which its
package list never contained.

**Baseline, CURRENT as of 2026-08-23:** `6756f9131`, `ctest -j1 --timeout 1800
--output-on-failure` inside an `rc run -d thor:gpu0` lease, 632.35 s — **598
tests, 573 passed / 3 skipped / 22 red.** Job
`8bf39567-9334-4f7e-aa27-43a2aa867bb7`, artifacts under
`/mnt/nas_share/rc/thor-w05-955/out/`. **The [#955](https://github.com/mudler/vllm.cpp/issues/955)
staleness debt is PAID**: the previous baseline `0764ded2b` was 176 commits
behind, 123 of them touching `src/`, `include/`, `tests/` or `CMakeLists.txt`,
384 files and +91,929 lines.

**Three things the re-measurement settled.** Every `SEGFAULT` on the box is
gone — `test_capi`, `test_ops_fp8_cutlass` and
`test_ops_matmul_fp8_block_cuda` all improved to `Failed`, and the prediction
this spec recorded was right: `cffe59b02` made the portable reference tier
ineligible on a backend that does not report its device memory host-addressable,
which is exactly Thor.
[#1725](https://github.com/mudler/vllm.cpp/issues/1725) is therefore half
resolved and was RE-SCOPED rather than closed on 2026-08-23, because the
block-scaled op still refuses generically instead of by name.
[#962](https://github.com/mudler/vllm.cpp/issues/962) did NOT move: the NVFP4
marlin self-disagreement reproduces byte-identically at `bitdiff=15/32768`.
And `test_serve_low_tools` is **no longer** the absent `shellcheck` of
[#961](https://github.com/mudler/vllm.cpp/issues/961) — `73ada0df8` fixed that
guard, and the entry now covers four unrelated `test_dflash2_speed_harness.py`
cases, proved by re-running the test with `shellcheck` 0.9.0 installed and
getting the identical four. Six names arrived with no owner; they and the two
stale-cause entries are [#1802](https://github.com/mudler/vllm.cpp/issues/1802).

**The cubin proof is no longer owed.** With `cuobjdump` installed and asserted,
33 `*.cu.o` objects carry 33 `sm_110` cubins, one each — which retires the
unverified 2026-08-15 claim of "30 objects, one `sm_110` cubin each".

The W0.5 gate as originally written ("the existing suite passes there") is not
met and was the wrong gate; §7 records the differential gate that replaces it
and the reasoning for it. That gate is keyed on `(name, failure mode)` pairs,
not on the length of the list — the version that shipped counted names, and
five `Failed` → `SEGFAULT` transitions prove counting names is too weak. The
lane earned its keep immediately by finding
[#960](https://github.com/mudler/vllm.cpp/issues/960) — an FP8 refusal on `main`
that became a silent portable-CPU fallback and a segfault, invisible on GB10 —
and it earned it again on 2026-08-19, when the same three tests came back as
`Failed` rather than `SEGFAULT` and a name-counting gate would have seen
nothing.

**W1 — DONE.** `Dots3NoteForCausalLM` resolves through the registry, parses the
released `config.json`, and accounts for 1614/1614 tensors on a committed slice
of the released shard index; load, GGUF and the forward each refuse by name.
All six §4 traps carry RED-before and green-after evidence in §4.1, which also
records four things W0 did not know: §1.4 is resolved (the checkpoint ships
exactly one nextn layer), that nextn block carries the SLIDING geometry with a
DENSE MLP, `mlp.gate.e_score_correction_bias` ships F32 in an otherwise BF16
tower, and §4 item 6's `is_neox_style` reading was wrong
([#1804](https://github.com/mudler/vllm.cpp/issues/1804), corrected in place).

**The row stays `SPIKE`, deliberately.** Making an architecture resolve is not
porting a model, and the §8.1 heading restructure that `ACTIVE` requires belongs
to the brick where the forward stops refusing. `MODEL-MM-muse-glimmer-*` records
the same reasoning with a whole text forward landed.

**W2 — DONE 2026-08-24.** The whole `model.safetensors.index.json` rather than
W1's four-layer slice: **38006/38006 accounted**, 35381 language / 2195 vision /
430 audio, every bucket asserted by number, zero unaccounted. Headers only —
4770592 bytes over the 133 shard headers, no tensor byte, no GPU. The two tower
files are now NAMED DEFERRAL RECORDS rather than integer counters, and the load
refusal prints the table. §4.4 carries the evidence, the fetch recipe and the
mutation table.

**W2 settled three things W1 could only claim, and found a fourth.** The
backbone has exactly four distinct layer shapes, so the 1/2 repeat holds and no
layer breaks it. The full/sliding split read off the shipped indexer tensors
equals `config.layer_types` exactly. The checkpoint carries 62 F32 tensors in
TWO families — the 45 language `e_score_correction_bias` W1 predicted, plus 17
`vision_encoder.blocks.{25..41}.mlp.router_bias` its language-only slice could
not see, which is spec R5's shape. And the index declares
`indexer_rope_layout: "leading"` / `indexer_rope_converted_from: "tail"`, which
NO upstream code reads: [#1846](https://github.com/mudler/vllm.cpp/issues/1846),
owed by W3.

W10 still owes one reconciliation neither W1 nor W2 could make: upstream's
`config.layer_types[layer_idx]` has no entry at the nextn index, so the
checkpoint — not `model.py:503` — is what says that block is sliding.

**W3 — DONE 2026-08-25.** The first maths this row has written:
`_forward_note_mla`'s full-attention arm as a portable HOST reference
(`dots3_note_attn.{h,cpp}`), with all four deltas over plain DeepSeek MLA and
the DSA indexer's rope geometry. **Gate met: `test_dots3_note_attn`, 12 cases /
198 assertions, CPU-only, no GPU, no checkpoint**, against an independent
double-precision reference transcribed from the upstream python — a complex
rotation, a max-subtraction-free `long double` softmax, a full-sort top-k — that
agrees with the implementation to 1.7e-16 to 3.2e-16 relative on every traced
intermediate. RED-first: with all four deltas neutralised the same gate reads 4
cases / 12 assertions failed. §4.5 carries the anchors, the properties, the
19-row mutation table and the two instrument defects the RED arm found.

**W3 discharged #1846** — the indexer rotates the LEADING 64 lanes of the
128-wide index head, which is the released index's declared
`indexer_rope_layout` and what `deepseek_v2.py:804-805` does; the tail slice
moves 10 of 24 selection slots. It is a DIFFERENT question from §4 trap 2's
GPT-J/NeoX pairing, which moves 7, and the gate shows the two disagree with each
other, so neither subsumes the other.

**One mutation came back GREEN and the code changed rather than the table.**
Feeding the indexer the unrescaled `q_c` changes nothing, because a positive
rescale multiplies every logit in a row by one constant and the argmax does not
move — so §4 trap 5 reaches the output through the MLA scores and through
nothing else. A comment that claimed otherwise is corrected and the invariance
is now asserted.

**The row stays `SPIKE`, and the reason moved again.** It is no longer "no maths
has been written"; it is that the maths is not reachable. Nothing in
`ModelRegistry::Forward` touches the new code, `Dots3NoteModel::ForwardDevice`
still refuses by name, and the last case of the W3 gate asserts that refusal so
the boundary is executable. `## Owed` names the wiring, the brick that owes it
and the issue.

**Next dispatchable: W4 — sliding-window MLA.** It inherits W3's two named
debts: the device wiring of the full arm through the padded sparse MLA backend,
and the `mla::ForwardMlaAttentionBlock` extension that three of W3's four deltas
need.
