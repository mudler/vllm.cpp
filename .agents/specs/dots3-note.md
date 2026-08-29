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
**That directive names the END-TO-END host, and it does not reach the FA-2
path.** Thor's sm_110 is outside `VT_CUDA_FEATURE_TABLE`'s `fa2` row, so
`MlaPrefillAttentionCuda` throws there rather than computing; W4b-2's two
windowed CUDA kernels were compiled and executed on `orin:gpu0` (sm_87) instead,
through an `rc` lease on 2026-08-26 (§4.8). Read the `fa2` row before booking a
lease for anything on the FA-2 path, and pick the host by capability.
**Status:** W5 — **the RELEASED `config.json` is REPRESENTABLE for the first
time** (§7 W5, evidence §4.10). `Dots3NoteDeviceRefusal` returns "" for
`dots-studio/dots3-note-prev`: W5 put the 45 MoE layers on the decode path
through `Dots3NoteMoeBlock` over the shared `RunMoePlaced` seam, and W5c
([#2176](https://github.com/mudler/vllm.cpp/issues/2176)) removed the nextn
branch, which was STRICTER THAN UPSTREAM. **Representable is not runnable**: the
MoE alone is 545.82 GB of a 576.89 GB checkpoint (94.62%), so nothing this
project owns can feed it, and the 298.67 GB fp8 sibling is refused BY NAME as
W9.
That sits on top of W4b-3c — **the DSA lightning indexer's SELECTION is on the
decode path** (§7 W4b-3, evidence §4.9), on W4b-2's two attention geometries
(§4.8), W4b-1's host maths (§4.7), W4a's full-attention layer (§4.6), W3's host
reference (§4.5), W2's whole weight map (§4.4) and W1's config + registry
(§4.1). The arch RESOLVES, parses, accounts for 38006/38006 of the released
checkpoint's tensors, and DECODES a config whose layers are any mix of
`full_attention` and `sliding_attention` with dense MLPs — through
`ModelRegistry::Forward`, over an `mla::ForwardMlaAttentionBlock` that carries
dots3-note's two LoRA rescales, its `k_rope_only_layernorm`, its headwise gate
and its 513-wide window, reading a PADDED 1088-wide MLA cache row narrowed to
each layer's own logical width, and that now computes the indexer's logits and
its top-k and attends only the selected slots. A step past `index_topk` whose
requests are all single-shot prefills is SERVED sparsely; a step in which any
request resumes is REFUSED BY NAME, because the indexer's own key cache is
`KV-DSV4-MULTICACHE` ([#1925](https://github.com/mudler/vllm.cpp/issues/1925))
and not this row. The RELEASED checkpoint still REFUSES BY NAME, now at its
first MoE layer until W5 LIFTED it, and both towers still do. Exactly ONE GPU lease has
run a BRICK GATE of this row, at kernel level and no further: `orin:gpu0`
(sm_87) compiled and ran W4b-2's two windowed CUDA ops on 2026-08-26 (§4.8).
The row's other leases were `thor:gpu0` provisioning and `ctest` baseline runs,
which gate the HOST and not this model (§7 W0.5). No brick has run the MODEL on
a GPU, and no tensor byte of the checkpoint has been downloaded: the committed
fixtures are the released `config.json` and a headers-only projection of the
complete shard index. The row stays `SPIKE`.

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
`${VLLM_SOURCE}`); the absence of the architecture at our pin; the live state
of Thor in §6.3 (one read-only `ssh` probe); and, on `orin:gpu0` under an `rc`
lease, the compilation and on-device execution of W4b-2's two windowed CUDA ops
(§4.8). The last of these is kernel-level parity on two ops. It is not a
measurement of this model.

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
The `-fp8` sibling is 298,673,280,504 bytes = **298.67 GB**.

**SIZES IN THIS SPEC ARE DECIMAL GB (10^9 bytes)**, which is the unit the HF API
returns and the unit the 576.89 / 545.82 / 543.58 GB figures below are already
in; a binary figure is always written `GiB`. W0 through W5 wrote the fp8 sibling
as "~290 GB", which is neither convention — 298,673,280,504 B is 298.67 GB
decimal or 278.16 GiB binary. This is the one brick whose whole honesty argument
rests on exact sizes, so the measured number is quoted rather than rounded.

There is no smaller `dots3-note` variant in the
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

`vllm/models/dots3_note/nvidia/attention.py`, 815 lines, and the reason this row
is not a mechanical port:

- `_gather_swa_kv_kernel:49` — Triton kernel gathering the windowed KV rows.
- `_apply_swa_score_mask_kernel:119` — the window mask over scores.
- `_build_sliding_window_metadata:192`, `_SlidingWindowChunk:172`,
  `_SlidingWindowMetadata:187` — chunked window bookkeeping.
- `Dots3NoteMLAMetadataBuilder(TritonMLAMetadataBuilder):307`, including
  `_reserve_attn_logits_workspace:316`.
- `Dots3NoteTritonMLAImpl(TritonMLAImpl):439` with `_forward_swa_mqa:470`,
  `forward_mha:565`, `forward_mqa:656`.
- `Dots3NoteFlashAttnPrefillBackend:258` with `run_sliding_window:279`.
- `Dots3NotePaddedSparseImpl(FlashAttnMLASparseImpl):697` — the full layers,
  with `_logical_cache:700` narrowing the padded physical row back to the logical
  576 and `do_kv_cache_update:704`.

**EVERY anchor in that list was stale, and they are corrected in place at W4b-1**
(`main` is never rewritten, so a wrong anchor is repaired where a reader hits
it). Re-derived at upstream `origin/main` = `d9fbe526c0`, whose
`vllm/models/dots3_note/` is byte-identical to the `06ecec7a84` W3 and W4a read,
so this is pure anchor rot rather than an upstream change — the same finding
§2.2 records, one section down. The file is 815 lines and W0 wrote 807.

The rot has TWO shapes and only one of them is obvious. Six of the seven bullets
were off by exactly ONE, uniformly one line EARLY — for four of them that is the
`@triton.jit` / `@dataclass` decorator rather than the `def` / `class` it
decorates, and for `_build_sliding_window_metadata` and
`Dots3NoteMLAMetadataBuilder`, which carry no decorator, it is a plain
off-by-one. Annoying, and a reader recovers. The `Dots3NotePaddedSparseImpl` family
was off by EIGHT, and that one is dangerous in the way §2.2's items 1 and 2
were: `:689` is `get_name` on a DIFFERENT class (`Dots3NotePaddedSparseBackend`,
which really starts at `:678`), `:692` is a bare `@staticmethod` decorator, and
`:696` is a blank line between the two classes. A reader checking the first
finds plausible, related code on a neighbouring class and moves on. `check-symbol-
anchors.py` cannot catch this — its own docstring says it does not verify line
citations ([#1139](https://github.com/mudler/vllm.cpp/issues/1139)).

The padding contract lives in `model.py:204-216`: `Dots3NotePaddedMLAAttention`
overrides `get_kv_cache_spec` to report `physical_head_size` (`:216`), which
`Dots3NoteFullAttention` passes as `swa_kv_lora_rank + swa_qk_rope_head_dim`
(`:283`), so full and sliding layers allocate the same block while each reads its
own logical width.

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

`nvidia/multimodal.py:65` `Dots3NoteForCausalLM(nn.Module, SupportsMultiModal,
SupportsPP)` — `get_placeholder_str:81`, `_process_image_input:173`,
`_process_audio_input:186`, `_process_video_input:202`, `embed_multimodal:255`,
`get_mm_mapping:325`. `common/processor.py` (811) and `common/video.py` (497)
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

### 4.6 W4a put the full arm on the decode path, and extended a SACRED seam to do it

**LANDED at W4a** (`row/MODEL-MM-dots3-note-W4a`,
`include/vllm/model_executor/models/mla_attention.h` +
`src/vllm/model_executor/layers/attention/mla_attention.cpp` (the seam),
`src/vllm/model_executor/models/dots3_note_device.cpp` (the wiring),
`tests/vllm/models/test_dots3_note_attn.cpp` and
`tests/vllm/model_executor/layers/attention/test_mla_attention_block.cpp` (the
gates), upstream re-read at vLLM `origin/main` `06ecec7a84`). CPU-only. No GPU
lease was taken and none was needed.

**Every number in this section is from the head that landed**, re-derived after
the fresh review's findings changed six of them. The first draft carried
pre-sharpening figures that no longer reproduced, which is review finding F3 and
is worth naming because §4.6 is the surface a re-runner checks against.

**W4 IS SPLIT, and this section is the first half.** §7's W4 bullet already said
"the largest brick; likely splits further". W4a is exactly the two items W3 left
under `## Owed`: the `mla::ForwardMlaAttentionBlock` extension, and the full arm
on the decode path. **W4b keeps everything the sliding geometry needs** — the
windowed metadata, the KV gather, the score mask and the padded/heterogeneous KV
spec of §2.3 — and it also inherits the DSA sparse SELECTION on device, which
W4a refuses by name rather than approximating.

#### What the seam grew, and why that shape

W3's own comment argued the extension was three fields; the W3 review argued four
of four deltas were unrepresentable. Both were reasoning about a HOST, cache-free,
`double` reference, which cannot route through a device seam at all. With a
device forward as the goal the question changes, and the answer is four fields:

| field | default (= ABSENT) | upstream | why the SEAM owns it |
|---|---|---|---|
| `MlaBlockDims::q_lora_scale` | `1.0` | `model.py:155` | it lands between `q_a_layernorm` and `q_b_proj`, both inside the block |
| `MlaBlockDims::kv_lora_scale` | `1.0` | `model.py:159` | it lands between `kv_a_layernorm` and the cache write, both inside the block |
| `MlaBlockWeights::k_rope_only_layernorm` | empty | `model.py:160`, built `:299-301` | it lands between the kv A-projection and the decoupled RoPE, both inside the block |
| `MlaBlockWeights::attn_gate_proj` | empty | `model.py:190-197`, built `:286-298` | it lands between the attention output and `o_proj`, both inside the block |

`double` and not `float` for the two scalars because upstream's value is a python
float and the rounding to the activation dtype has to happen once, in the op.
The two weights are `vt::Tensor`, so EMPTY is their absent state and a null
pointer is what the branch tests.

**Absence is not a no-op, it is a NOT-TAKEN branch.** At `1.0` the `vt::MulScalar`
is not launched; with the weights empty no buffer is allocated, no GEMM is issued
and no gate kernel runs. The "no buffer" half of that was FALSE in the first
draft and is review finding F4: the gate's scratch buffers were constructed
unconditionally at zero width, and `dense_device_glue.h:118` rounds a zero-length
request up to one byte, so each still took and returned a pool block per layer
per step on the SACRED path. They now live in a `std::vector<DBuf>` that stays
empty. No numeric effect either way — but the sentence is load-bearing in the
byte-identity argument below, so it had to become true rather than be softened.

Four consequences are recorded rather than hidden:

1. **`k_rope_only_layernorm` turns OFF the Tier-A2+A5 fused-norm-rope fold.**
   `vt::FusedNormRope` ropes `k_pe` straight out of the merged `[L+R]` kv row and
   has no step in which the rope half could be normalized first, so its presence
   takes the split path. DeepSeek never sets the weight and keeps the fold.
   Mutation **M8** is that guard: remove it and the norm is silently skipped.
2. **The headwise gate requires the block to run in BF16, and refuses at ENTRY.**
   It is realized with `vt::SharedExpertGate` over the `[T*num_heads,
   v_head_dim]` view of the attention output — the same per-row sigmoid
   broadcast Qwen3.6's shared-expert gate already ships — and that op stores
   bf16 only. bf16 IS upstream's activation dtype for this model
   (`porting.md`: one model dtype), so the refusal costs nothing real. It fires
   at function entry rather than in step 5c because step 4 has by then written
   this token's K/V into the paged cache, and a throw after that leaves the
   cache MUTATED for a request that produced no output — review finding F6.
3. **The gate LOGIT is bf16, and that was wrong in the first draft.** Upstream
   builds `g_proj` with no `params_dtype` (`model.py:292-297`), so it inherits
   the model dtype and the sigmoid input is a bf16 value that `.float()` widens.
   The first draft emitted an f32 GEMM output — strictly WIDER than upstream on
   a model path, which is the AGENTS.md clause a token gate cannot enforce.
   Review finding F2. The GEMM now stores bf16 and `vt::CastF32` widens it
   exactly, which is upstream's `.float()`.
4. **ONE rounding step remains unmirrored, and it is the last one.** Upstream
   narrows the SIGMOID to bf16 before multiplying (`model.py:196-197`), so its
   product is rounded twice; `vt::SharedExpertGate` keeps the sigmoid in f32 and
   rounds only the product. Fewer roundings, strictly closer to the real value,
   and byte-for-byte the convention this tree already ships for the
   shared-expert gate. Bounded below.

#### The DeepSeek-V2 path is byte-identical, MEASURED across the seam's callers

`mla::ForwardMlaAttentionBlock` has FOUR callers — `deepseek_v2`, `minicpm3`,
`kimi_linear` and now `dots3_note` — and DeepSeek-V2-Lite carries a SACRED
token-exact gate. "Defaulted fields cannot change the old path" is an argument,
not evidence, so the argument was replaced with a fingerprint.

A scratch probe (NOT committed — it is measurement scaffolding, like a mutation)
ran six fixed batches through the block and printed an FNV-1a64 over the RAW
output bytes. The BEFORE arm is a separate `git archive` tree at the base SHA
`d7d1ee914` with the byte-identical probe appended, its own `cmake` configure and
its own build — not a stashed file in this tree.

##### The stale-binary false confirmation, which is the transferable part

That distinction is not pedantry, and the reason is the most reusable thing this
brick produced. The FIRST attempt at the BEFORE arm reverted only the two seam
files in place and rebuilt. What happened next:

1. the build FAILED — the dots3 TU still referenced `q_lora_scale`,
   `kv_lora_scale`, `k_rope_only_layernorm` and `attn_gate_proj`, which the base
   header does not declare;
2. `cmake --build` exited non-zero, but the PREVIOUS binary was still on disk;
3. running it printed six fingerprints;
4. **all six matched the head exactly — because they WERE the head's.**

Read without checking the compile status, that is a perfect proof of
byte-identity. It is also a measurement of nothing at all. The failure mode is
worse than a wrong number: a wrong number invites a second look, and this one
agrees with the hypothesis.

**The rule this hands to every later brick: a mutation or A/B harness must treat
a NON-ZERO COMPILE EXIT as `NOT A RESULT`, never as a run, and must print that
exit beside every row.** This row's own mutation driver already does — which is
why the 18-row table below carries a `compiler exit` column and why W2's spec
says the same thing — but the A/B probe was driven by hand and had no such
guard, so the discipline that protected the mutations did not protect the
measurement standing beside them. The structural fix is the one applied here: an
independent `git archive` tree at the base SHA, its own configure, its own build,
so there is no previous binary to fall back to. It is the same family as
[the stale-binary and incremental-build notes](../verification.md) and as W2's
exit-135 finding, where a crash also read as "no result" rather than as failure.

A base measurement has to come from a build that succeeded, and the harness has
to be the thing that knows it.

The six arms cover the seam's whole branch space — q_lora present/absent, both
rope layouts, both dtypes — rather than six variations of one caller:

| arm | geometry | bytes | BASE `d7d1ee914` | HEAD |
|---|---|---:|---|---|
| 0 | V2-Lite, f32, MIXED (2 decode + 2 prefill, one with context) | 106496 | `2071435139082975929` | identical |
| 1 | V2-Lite, bf16, same batch | 53248 | `15607516550467795365` | identical |
| 2 | V3 q_lora branch, f32 | 86016 | `4982522374592074643` | identical |
| 3 | V3 q_lora branch, bf16 | 43008 | `3757253798370478450` | identical |
| 4 | MiniCPM3 (`is_neox_style=true`), f32 | 30720 | `9024916185557934982` | identical |
| 5 | MiniCPM3 (`is_neox_style=true`), bf16 | 15360 | `16077001697345918067` | identical |

Six for six. **Arms 2-3 are in the table because `q_lora_scale` inserts into
that branch and nowhere else**, and **arms 4-5 because `is_neox_style` is the
only MLA-geometry field that differs between the seam's families**, so a probe
that only ran DeepSeek would have proved nothing about MiniCPM3. Kimi-Linear
takes the same branches DeepSeek does and adds no coverage; that is stated rather
than padded with a seventh arm.

**Both DeepSeek gates were also run at the base SHA and at this head.**
`test_mla_attention_block` reads 10 cases / 2247703 assertions at the base and
**12 / 2247715** here: the two extra cases are this section's own seam-contract
cases, and every one of the 2247703 pre-existing assertions is unmoved.
`test_deepseek_v2_forward` reads **11 cases / 1052 assertions on both sides,
identical** — which is also the evidence for the `MlaStep`/`BuildMlaStep` move,
since that function is what its CPU synthetic forward drives.

**NOT run, and named rather than implied:** the SACRED e2e token gate itself.
It needs a DeepSeek-V2-Lite checkpoint on a CUDA host, and this brick ran
CPU-only on a box with no GPU. What IS run here is the same block, the same
batch shapes and the same weights the SACRED gate decodes through, byte-compared
on six geometries.

#### What the wiring is, and the one config shape it covers

`Dots3NoteModel::ForwardDevice` is a real forward now: embed →
{`input_layernorm` → `mla::ForwardMlaAttentionBlock` → `post_attention_layernorm`
→ dense SwiGLU MLP} per layer → final norm → `lm_head`, over the shared
`BuildMlaStep` metadata build, with the three residual add+RMSNorm sites routed
through `vt::FusedChain`. It covers **one** config shape — every layer
`full_attention` with a DENSE MLP — and refuses everything else BY NAME:

| refused | where the refusal lives | brick |
|---|---|---|
| any `sliding_attention` layer | `Dots3NoteDeviceRefusal`, config | W4b |
| any MoE layer | `Dots3NoteDeviceRefusal`, config | W5 |
| a PADDED physical latent row | `Dots3NoteDeviceRefusal`, config | W4b |
| a nextn tail | `Dots3NoteDeviceRefusal`, config | W10 |
| a request whose `seq_len` exceeds `index_topk` | the forward, per step | W4b |
| a KV cache row disagreeing with the config | the forward, per step | W4b |
| the vision / audio towers | the LOADER's deferral table (§4.4) | W6 / W7 |

**The last three rows are the shape of review finding F5.** The first draft put
the padded-row and nextn checks only at the forward, or nowhere, and claimed the
towers among them. The consequence was real rather than cosmetic: the loader
materialized a whole tower for a config the very next call refused. The two
config-level checks moved into `Dots3NoteDeviceRefusal`, which is the predicate
the loader itself consults, and the per-step checks stayed where only a per-step
input can reach them. The cache-row check is now REACHED by a case that hands
the forward a deliberately wrong row, so it is gated rather than defensive
decoration.

**The released `dots-studio/dots3-note-prev` config still refuses**, at layer 1
(MoE) and layer 2 (sliding), so nothing a user can run changed and
`test_dots3_note_scaffold` is unmoved at 26 cases / 110818 assertions.

**Materialization is conditional on that same predicate, and the reason is a
measurement.** `LoadDots3NoteWeights` materializes only when
`Dots3NoteDeviceRefusal` is empty. Materializing unconditionally was rejected
because the released config's `embed_tokens` alone is 152064 x 5120 bf16 =
1.5 GiB while W1/W2's gate drives the whole 38006-name index through this loader
from a synthetic checkpoint of ONE-ELEMENT tensors; demanding real shapes there
would either red the accounting gate or require a fixture nothing can hold.

**`MlaStep` / `BuildMlaStep` moved out of `deepseek_v2.cpp`'s anonymous
namespace** into `deepseek_v2.h`. Upstream's own class is
`Dots3NoteFullAttention(DeepseekV2MLAAttention)` and runs the same block over the
same paged MLA cache, so a second copy of the metadata build in another TU is the
hand-rolled parallel path AGENTS.md forbids. Nothing in the body changed; the
byte-identity table above and both DeepSeek gates are the move's evidence.

#### The gate, met

`test_dots3_note_attn` — **18 cases / 638 assertions**, CPU-only, no GPU, no
checkpoint, no speed claim (12/198 at W3). Six new cases drive the DEVICE path
through `ModelRegistry::Resolve(config)` → `reg.factory->load_weights` →
`ModelRegistry::Forward`, over a REAL synthetic safetensors checkpoint with real
shapes, and compare against a whole-model double reference built on W3's
`ref::Forward` — the same independent transcription, unchanged except for two
switches this brick needed (the two LoRA rescales became independent flags, and
the gate logit gained a width switch).

**What the comparison can and cannot say.** The device arm stores every
activation in bf16, which is upstream's model dtype; the reference is double
throughout. The residue is therefore a bf16 quantisation floor and not a
mechanism difference, and the case PRINTS it: **max|diff| 0.05268 over a scale of
2.951 = 0.0178515 relative**. It is a consistency gate, not a correctness gate —
§6.4 option B, no vLLM oracle exists for this model on any host we own.

**The BOUND is chosen for separation, and that is review finding F1.** The first
draft declared `2e-2` with the residue at 1.9e-2 and a seam mutation dropping
`q_lora_scale` reddening at 2.0952e-2 — a **4.8% margin** on the single field
that touches the DeepSeek-V3 q_lora branch, one seed or one compiler from a
false green. The cause was fixture geometry, not the number: at `q_lora=6,
kv_lora=4` over `hidden=8` the two scales were 1.155 and 1.414, against the
released model's 2.236 and 3.162. The bench ranks are now `q_lora=3, kv_lora=2`
over `hidden=16`, giving 2.309 and 2.828 — the released ratio's neighbourhood,
still different from each other so a swap cannot hide, and `kv_lora >= 2` so the
latent RMSNorm is not the degenerate 1-wide one.

The bound is **5e-2**, and the three ratios it sits between are kept SEPARATE:

| ratio | value | what it says |
|---|---:|---|
| bound / residue | 0.05 / 0.0179 = **2.8x** | headroom above the bf16 floor |
| nearest mutation / bound | 0.761 / 0.05 = **15.2x** | headroom below the nearest defect — the number that says this bound cannot admit a missing `q_lora_scale` |
| nearest mutation / residue | 0.761 / 0.0179 = **42.6x** | separation of the whole instrument, a statement about the FIXTURE and not about the bound |

**A draft of this section wrote "2.8x above the residue and 43x below the
nearest mutation", which merged rows one and three and overstated the
bound-to-mutation headroom by 2.8x.** The review caught it. It is the same class
of error as F1 itself — a ratio that reads like margin and is measuring a
different pair — so the table stays in place of the sentence.

**Each of the four new fields is shown to be EXERCISED, not merely compiled**, by
neutralising it in the REFERENCE and measuring the device arm drifting AWAY:

Both ratio columns are given, and labelled, for the reason the bound table
above states: the two are different statements and one of them was quoted as the
other in an earlier draft.

| the reference with … | device-vs-reference | / the 5e-2 BOUND | / the 0.01785 RESIDUE |
|---|---:|---:|---:|
| the **q** LoRA rescale dropped | 0.760958 | 15.2x | 42.6x |
| the **kv** LoRA rescale dropped | 1.00598 | 20.1x | 56.4x |
| both LoRA rescales dropped | 0.811884 | 16.2x | 45.5x |
| `k_rope_only_layernorm` dropped | 0.995095 | 19.9x | 55.7x |
| the headwise gate made lane-wise | 0.889367 | 17.8x | 49.8x |

**The two scales are neutralised SEPARATELY as well as together**, which is the
other half of F1: an arm that drops both at once cannot distinguish a port that
carries both from one that dropped only the q scale, and the combined figure
(0.812) is not even the largest of the three, so it is not a conservative stand-in.

The `k_rope_only_layernorm` fixture had to be BUILT to make that mechanism
observable, and that is stated rather than tuned away. RoPE preserves the L2 norm
of each rotated pair exactly, so the norm's ORDER commutes with the rotation
whenever `w_{2i} == w_{2i+1}`, and only the per-lane weight fails to commute. At
`TinyWeights`' defaults those weights hug 1.0 and mutation M5 slipped UNDER the
bound. The norm's weight now alternates 2.5 / 0.3 within each rotated pair, which
is the minimal targeted fix, and `CHECK_FALSE(md.is_neox_style)` anchors it: a
future flip to half-split pairing reds the geometry assertion first.

**The gate's TWO widths are answered, and only one of them by a gate.** W3's
double reference could see neither.

- **The logit.** Mirrored (bf16, per point 3 above). No value gate on a bf16
  output can confirm that, and the case says so with a number rather than a
  shrug: rounding the logit moves the gate by at most
  `max_x[σ(1-σ)|x|] · 2⁻⁹ = 0.2239 · 2⁻⁹ = 4.38e-4` absolute, measured at
  **3.715e-4** here, while the gated product's own bf16 store has a half-ulp of
  **1.953e-3**. The signal is under the floor by construction. Mutation **M16**
  reverts the narrowing and comes back **GREEN**, which is that analysis
  executed rather than argued. The relative form is deliberately NOT claimed as
  bounded — at `x → -∞` the gate vanishes while `|dσ/σ| = (1-σ)|x|` grows without
  limit — and a first draft that asserted a relative bound was wrong for exactly
  that reason, having scanned only positive logits. It is reported as measured
  (1.657e-3) instead.
- **The sigmoid.** Not mirrored, and bounded: `|bf16(σ) - σ| ≤ 1.899e-3`, under
  the analytic `2⁻⁹ = 1.953e-3` that holds for any fixture because a sigmoid is
  in (0,1); and the extra rounding upstream applies moves the gated output by
  **3.906e-3 over a scale of 0.9453**, i.e. under `2⁻⁷` relative. That last
  figure is EMPIRICAL and now says so — the first draft presented it as a bound.

#### The mutation table

Every mutation was applied to the tracked source, rebuilt, run, and reverted,
with the tree verified byte-for-byte afterwards (18 of 18 restored). **The
compiler exit status is printed beside each row**, because a mutation that fails
to build reads as a passing test and this project has been bitten by that
repeatedly. `cases`/`assertions` are what `doctest` reported FAILING.

| id | mutation | compiler exit | result | cases | assertions | first failing case |
|---|---|---:|---|---:|---:|---|
| R0 | RED-FIRST: all four new seam fields neutralised AT ONCE | 0 | RED | 2 | 6 | W4a: REACHED through `ModelRegistry::Forward` |
| M1 | `q_lora_scale` is never applied | 0 | RED | 2 | 6 | W4a: REACHED through `ModelRegistry::Forward` |
| M2 | `kv_lora_scale` is never applied | 0 | RED | 2 | 6 | W4a: REACHED through `ModelRegistry::Forward` |
| M3 | the q rescale moves BEFORE its layernorm | 0 | RED | 2 | 6 | W4a: REACHED through `ModelRegistry::Forward` |
| M4 | `k_rope_only_layernorm` is dropped | 0 | RED | 2 | 6 | W4a: REACHED through `ModelRegistry::Forward` |
| M5 | `k_rope_only_layernorm` is applied AFTER the rope | 0 | RED | 2 | 6 | W4a: REACHED through `ModelRegistry::Forward` |
| M6 | the headwise gate is skipped | 0 | RED | 2 | 6 | W4a: REACHED through `ModelRegistry::Forward` |
| M7 | the gate reads the POST-attention state, not the layer input | 0 | RED | 2 | 6 | W4a: REACHED through `ModelRegistry::Forward` |
| M8 | the fused-norm-rope fold is NOT disabled by the k_pe norm | 0 | RED | 2 | 6 | W4a: REACHED through `ModelRegistry::Forward` |
| M9 | REACHABILITY: the production materialization CALL SITE is deleted | 0 | RED on `test_dots3_note_attn`, **GREEN on `test_dots3_note_scaffold`** | 5 / 0 | 3 / 0 | W4a: REACHED through `ModelRegistry::Forward` |
| M10 | the scope refusal accepts EVERY config | 0 | RED | 1 | 4 | W4a: what the device path still REFUSES, by name |
| M11 | the `index_topk` refusal is deleted | 0 | RED | 1 | 1 | W4a: what the device path still REFUSES, by name |
| M12 | the seam's two scales stop being resolved from the config | 0 | RED | 2 | 11 | W4a: REACHED through `ModelRegistry::Forward` |
| M14 | the load-time shape check on `g_proj` is deleted | 0 | RED | 1 | 1 | W4a: a weight of the WRONG shape refuses BY NAME at load |
| M15 | the `!= 1.0` LAUNCH guards are removed | 0 | **GREEN** | 0 | 0 | — measured on `test_mla_attention_block` + `test_deepseek_v2_forward` |
| M16 | F2 reverted: the gate logit GEMM stores f32 again | 0 | **GREEN** | 0 | 0 | — the width analysis above, executed |
| M17 | the PADDED-latent-row refusal is deleted | 0 | RED | 1 | 2 | W4a: what the device path still REFUSES, by name |
| M18 | the nextn-tail refusal is deleted | 0 | RED | 1 | 2 | W4a: what the device path still REFUSES, by name |

There is no M13. The ids are the driver's and are left as they were RUN rather
than renumbered, because a tidy sequence is a smaller thing than a table a
re-runner can reproduce.

**R0 is the RED-first arm and it ran BEFORE the green one.** With all four
fields neutralised inside the seam — the two `vt::MulScalar` calls, the k_pe
norm and the whole gate block — the gate reads 2 cases / 6 assertions failing,
exit 1, compiler exit 0. That is a real run, not an inference.

**Two of the eighteen are not about the maths at all, and they are the ones that
say the code is REACHED.** M9 deletes the production materialization call site;
M12 stops `Dots3NoteFullAttnMlaDims` from reading the two scales off the released
params. Both go red, so the layer's geometry comes from `config.json` through the
real loader rather than from a struct the test typed.

**Three GREEN rows, and each says why.**

**M9's scaffold arm is the point of M9, not a miss.** Deleting the production
materialization call site takes `test_dots3_note_attn` RED and leaves
`test_dots3_note_scaffold` untouched — which is exactly right, and says which
gate reaches the new code: the scaffold drives the RELEASED config, which
refuses before any weight is read, so it cannot see a materialization that never
runs for it. The attn gate's five failing cases are the ones that do.

**M15 is a green this gate CANNOT close, and the reason is stated rather than
worked around.** It forces the q-branch guard open so `vt::MulScalar` runs with
a scalar of 1.0 on the DeepSeek path — and `test_mla_attention_block` does
exercise that branch, since its V3 case is the tree's only `q_lora_rank > 0`
coverage. Multiplying an IEEE float by 1.0 is exact, so every value is
unchanged, the byte-identity table above is unchanged, and both DeepSeek gates
stay green. The guards are therefore a LAUNCH-COUNT statement, not a value
statement: without the q guard every MLA model that HAS a q_lora branch pays one
extra kernel launch per layer for an identity, and without the kv guard every
MLA model pays another. That is a real cost on the SACRED path and a real reason
to keep them, and this tree has no op-invocation counter a doctest could assert
on — so the guards stay, with their green recorded here rather than in a comment
claiming a gate that does not exist.

**M16 is a green that was PREDICTED before it was run**, which is the only kind
worth having. The width case's analytic bound says a bf16 store cannot resolve
the logit's width; M16 reverts the narrowing and the gate does not move. The
narrowing stays because upstream's `g_proj` has no `params_dtype` and porting.md
requires the memory format to be checked against the source — not because a
number here would notice.

Contrast W2's M12, which deleted production code the gate could not see: that
code was UNREACHABLE. M15's and M16's code is reached and merely value-neutral.
Different findings, different answers.

**No regression on the sibling gates.** `test_dots3_note_scaffold` re-ran at this
head: 26 cases / 110818 assertions / 0 failed. `test_mla_attention_block`:
12 / 2247715 / 0 failed. `test_deepseek_v2_forward`: 11 / 1052 / 0 failed.
`test_deepseek_v2_decode_graph_seam`: 3 / 230 / 0 failed.

### 4.7 W4b-1 wrote the sliding maths, and W4b is SPLIT

**LANDED at W4b-1** (`row/MODEL-MM-dots3-note-W4b-1`,
`src/vllm/model_executor/models/dots3_note_attn.{h,cpp}`,
`tests/vllm/models/test_dots3_note_attn.cpp`, upstream re-read at vLLM
`origin/main` `d9fbe526c0`). CPU-only. No GPU lease was taken and none was
needed: a reference-versus-implementation gate has no device in it.

**W4b IS SPLIT, and this section is the first half.** §7's W4b bullet scoped the
whole §2.3 stack plus three device refusals W4a handed on. That is two bricks,
and the split line is the one this row already used once — W3 wrote the full
arm's maths as host code and W4a put it on the decode path. W4b-1 is the sliding
arm's maths and the §2.3 machinery as host code; **W4b-2 is that stack on the
decode path**, and it keeps all three refusals.

**The split holds, and the reason W4b-1 first gave for it was FALSE.** This
paragraph is corrected in place, because `main` is never rewritten and because
the wrong version is the more instructive one.

**What it claimed** was that `vt::ConcatAndCacheMla`, `vt::MlaDecodeAttention`
and the MLA prefill gather address the cache as CONTIGUOUS `[num_blocks,
block_size, head_size]`, so a padded row would be a change inside those ops on
both backends — a CUDA half unverifiable on a CPU-only box, on the SACRED
DeepSeek-V2 path.

**The fresh review of [#1949](https://github.com/mudler/vllm.cpp/pull/1949)
refuted it by EXECUTION, and this brick then reproduced the refutation rather
than accepting it.** All three ops source `stride[0]` and `stride[1]` FROM THE
TENSOR (`cpu_cache.cpp:99-100`, `cpu_mla_attn.cpp:99`,
`cpu_mla_prefill.cpp:180`), and `Tensor::Slice(2, 0, logical)` shrinks
`shape[2]` while KEEPING both strides (`tensor.cpp:80-84`) — which is precisely
upstream's `kv_cache[..., : self.head_size]`. The tree already GATES this:
`tests/vt/test_ops_mla_cache.cpp:259` is `TEST_CASE("concat_and_cache_mla is
STRIDE-driven (cache view + split sources)")`, with CUDA-vs-CPU strided parity
at `:403`. A scratch probe built a cache at physical row 7 / logical row 5,
wrote through `Slice(2, 0, 5)`, gathered back through it and DECODED through it,
asserting the two pad lanes untouched: **compiler exit 0, binary exit 0, 30/30
assertions, and ZERO changes to any `vt` op**. The only contiguous construction
is one model-level line, `dots3_note_device.cpp:470`.

**Two different reasons hold the two halves apart, and only one of them is a
constraint.**

- **The padded row is deferred BY SCOPE CHOICE.** It is expressible today and
  CPU-gateable. It is deferred because a padded row with no windowed attention
  to read it is half a capability rather than a shipped one, and because this
  pull request already carries a completed review. Deferring by choice with an
  accurate reason is fine; deferring behind a false constraint is the defect.
- **The WINDOW is the real constraint.** `vt::MlaDecodeAttention` attends over
  the WHOLE sequence — `for (int64_t j = 0; j < seq_len; ++j)`,
  `cpu_mla_attn.cpp:94` — with no window bound and no per-slot `valid`, and
  neither `MlaDecodeAttentionArgs` nor `MlaPrefillAttentionArgs` carries a window
  field at all. A windowed decode and prefill is therefore a NEW KERNEL on both
  backends, with a CUDA half no CPU-only box can verify, and it owes the seam
  byte-identity W4a produced for the four callers of
  `mla::ForwardMlaAttentionBlock`.

**Why the wrong version was worse than a wrong number.** It was committed in
five places, and a W4b-2 implementer reading it would have believed the shared
seam cannot express a padded row — which licenses editing three `vt` ops across
two backends, or hand-rolling the parallel path AGENTS.md forbids. That is the
same over-claiming-scope-statement class the W4a review already caught once on
this row.

**What W4b-1 does NOT do, stated before what it does.** No device path changed.
`Dots3NoteModel::ForwardDevice` still refuses a `sliding_attention` layer, a MoE
layer, a PADDED physical row and a nextn tail by name, and it still refuses a
request whose `seq_len` exceeds `index_topk`. **None of W4a's three refusals is
lifted here.** The last gate case asserts THREE refusals executably — MoE,
`sliding_attention` and the padded physical row — and exactly ONE of those three
is among W4a's three. The other two of W4a's, the `index_topk` bound and the
per-step cache-row check, stay executably asserted by W4a's own case, which this
brick did not touch. An earlier draft wrote "asserts three of them" with W4a's
three as the antecedent, which counted the same evidence twice. `## Owed`
records all of it against W4b-2.

#### What the sliding arm actually is

The two geometries are not a parameterisation of one another, which is why
`SlidingAttnDims` is its own struct rather than a flag on `FullAttnDims`:

| | full (13 layers) | sliding (33 layers) |
|---|---|---|
| heads | 128 | **64** |
| `kv_lora_rank` | 512 | **1024** |
| `qk_nope_head_dim` | 128 | **192** |
| rope theta | 8e7 | **5e4** |
| softmax scale | 192^-0.5 | **256^-0.5** |
| logical latent row | 576 | **1088** |
| window | — | **513** |
| DSA indexer | yes | **no** (`is_sparse = False`, model.py:432-434) |

Both ropes are GPT-J (§4 item 6), the headwise gate and `k_rope_only_layernorm`
are shared, and the physical KV row is 1088 for BOTH classes.

#### The four §2.3 mechanisms, and how each is reached

The layer does NOT compute a windowed copy of `ForwardFullAttention`. It
computes the attention the way upstream does — the ABSORBED MQA of
`_forward_swa_mqa` (`attention.py:470-563`) over a PAGED, PADDED latent cache —
so all four mechanisms are exercised by the layer's own comparison as well as by
their unit cases.

| Mechanism | Upstream | Reached by | The property, measured |
|---|---|---|---|
| the windowed metadata | `_build_sliding_window_metadata:192-254` | its own case | `kv_lens = min(seq_len, query_len + W - 1)` gathers **11** KV tokens where the three requests' full contexts are **34**; a port that gathered `seq_len` produces the SAME answer once the mask runs, at 3x the workspace, so no value assertion anywhere can catch it |
| the KV gather | `_gather_swa_kv_kernel:49-114` | the layer + its own case | a SHUFFLED block table `{2,0,1}` and a hand-derived expected slot per token; PAD INVARIANCE — the same logical rows in an unpadded cache gather byte-identically; a NEGATIVE block-table entry yields an invalid, zero slot |
| the score mask | `_apply_swa_score_mask_kernel:119-163` | the layer + its own case | a DECODE-shaped batch whose queries sit at positions 8/9 of a 10-long sequence, with the kept slot set derived BY HAND from `seq_len - query_len + q`; dense causal is a different answer by **0.786** relative and a window at `W == T` is the same one to **3.9e-16** |
| the padded row | `_logical_cache:700-702`, `get_kv_cache_spec:213-216` | its own case | the narrowing round-trips exactly, the tail of every physical row is UNTOUCHED by a logical-width write, and a reader that keeps the LOGICAL stride differs by **7.99** on the same buffer |

**The window's off-by-one has its own property**, because that is the defect a
port most plausibly ships and the one an output comparison alone can miss. At
`W == T` the windowed answer equals the dense causal answer to 3.9e-16; at
`W == T - 1` exactly ONE query loses exactly ONE key, and the layer moves by
**0.0836** relative. So the gate pins `kv_pos >= query_pos - W + 1` from both
sides rather than only asserting that a window does something.

#### The reference is independent, concretely

Same standard as W3, and the sliding arm needs it for the same reason: under
§6.4 option B there is no oracle and there will not be one at this stage. The
reference is transcribed from the python and it is a DIFFERENT ALGORITHM at four
levels, not a re-spelling:

- **materialized MHA** — `kv_b_proj` up-projects the latent into per-head K/V
  and the attention is an ordinary dot product, with no `W_UK`/`W_UV` fold and
  no latent-space intermediate. The implementation takes the ABSORBED route.
- **no cache at all** — key `s` is token `s`; no paging, no block table, no
  `slot_mapping`, no padded row, no gather.
- **the window is the direct positional predicate** `s <= t && t - s < W`, never
  `gather_start + slot` arithmetic.
- **softmax without the max subtraction**, in `long double`, and the rotation is
  a complex multiply with angles recomputed per element (W3's, reused: a
  reviewer proved that transcription independent in the STRONG direction by
  mutating the shared helper the IMPLEMENTATION routes through and watching the
  gate red — §4.5, R9).

The two arms agree to **1.2e-16 to 5.0e-16** relative on every traced
intermediate: `q_c`, `kv_c_normed`, `k_pe`, `q`, the attention output, the gate
and the layer output.

#### The gate, met

`test_dots3_note_attn` — **30 cases / 2418 assertions**, CPU-only, no GPU, no
checkpoint, no speed claim (18/638 at W4a, 12/198 at W3). Twelve new cases. The
geometry is resolved from the RELEASED `config.json` through
`ModelRegistry::Resolve(...)`, `factory->parse_config` and
`ParseDots3NoteParams`, never typed by hand — three reachability mutations say
so rather than the sentence (M23, M24, and M16/M18 on the two fields the
sliding arm alone carries).

**What the instrument measured, printed by the gate rather than assumed:**
`gather_len` 16 for a window of 3 over 8 tokens (a real round-up from 10, so 8
of the 16 gathered slots are past the sequence and come back invalid); **5 of
the 8 queries really lose a key**, 15 keys dropped in total; max |attention
score| 3.93, which keeps the reference's max-subtraction-free softmax inside
`exp`'s comfortable range.

**The agreement bound is `kAgreeRel = 1e-11`, inherited from W3 and NOT
re-argued**: both arms are double, the residue is pure reassociation at
1.2e-16 — 5.0e-16, and every mechanism difference below lands above 3e-2. The
three ratios are kept SEPARATE, because spec §4.6's review finding F1 is that
merging them overstates the headroom:

| ratio | value | what it says |
|---|---:|---|
| bound / residue | 1e-11 / 5.0e-16 = **2.0e4x** | headroom above the double reassociation floor |
| nearest mechanism / bound | 0.0300 / 1e-11 = **3.0e9x** | headroom below the nearest defect |
| nearest mechanism / residue | 0.0300 / 5.0e-16 = **6.0e13x** | separation of the whole instrument, a statement about the FIXTURE |

The nearest mechanism is the sliding-only one — the model-level rope theta
instead of `swa_rope_theta`, at 0.0300 relative. The other four sit at 0.485
(q rescale), 0.402 (kv rescale), 0.234 (`k_rope_only_layernorm`) and 0.439 (a
lane-wise gate). **The two LoRA scales are neutralised SEPARATELY and are
DIFFERENT numbers on this fixture** — `sqrt(16/3)` and `sqrt(16/6)` — so an arm
that dropped both at once could not distinguish a port carrying both from one
carrying only the q. Upstream's released ranks make the two sliding scales EQUAL
at `sqrt(5120/1024)`, and the fixture deliberately does not copy them; the
released-config case pins the released values separately.

**FIVE fixture separations are pinned, and they are pinned in the DELTAS case**
(`dots3-note W4b-1: the four deltas ... on the SLIDING arm too`), not in the
geometry case: `qk_head_dim != latent_row`, `num_heads != full_heads`, a
physical row wider than the full arm's logical one, `window < tokens`, and two
distinct rope thetas. An earlier draft of this section said four in one place
and three in another, and named the wrong case for both. **The head-count pin
was MISSING entirely until the fresh review**, and its absence is a measurement
rather than an oversight anyone argued about: setting `swa_heads` equal to
`full_heads` and changing nothing else left the whole gate green at 30 cases /
2417 assertions. With the pin added the same arm goes RED on 1 case / 1
assertion. Four of the five were written because a green mutation exposed the
geometry that hid a mechanism; the fifth is here because a reviewer went looking
for it and found no assertion behind it.

#### The mutation table

**The driver, named so the table is checkable from outside this document.** The
rows below were produced by a scratch driver at
`$SCRATCH/w4b/mutate.py` — measurement scaffolding, NOT committed, on the same
argument W4a's byte-identity probe records. It implements the four guards this
tree has paid for: it refuses an anchor that does not occur exactly once, prints
the COMPILER EXIT beside every row, rejects a binary older than the build, and
verifies the tree byte-for-byte after each restore.

**This tree already ships `scripts/mutation-harness.py`, which implements the
same four guards, and W4b-1 did not use it.** That is recorded rather than
glossed: it is a pre-existing tool from row LTX25-RES2S-LOOP
([#921](https://github.com/mudler/vllm.cpp/issues/921), landed at `4d7748646`)
and it is NOT this brick's work — the fresh review credited it to W4b-1, which
is wrong and is corrected here. Two rows cannot be expressed in it as it stands:
it applies exactly one find/replace per mutation, while **R0** needs four
simultaneous edits and the two review probes need edits in two files at once.
Every single-substitution row was CROSS-CHECKED through the committed harness
and agreed; the cross-check command and its output are in the paragraph after
the table.

**Every count below was RE-MEASURED after the head-count pin was added**,
rather than carried over and patched. Adding an assertion changes the gate, and
a table that mixed pre-pin and post-pin rows would be the failure this row keeps
naming: an instrument reporting on a state it was not given. Exactly one row
moved — `M24`, from 1 case / 2 assertions to 2 / 3, because the new pin is the
second thing it breaks.

Every mutation was applied to the tracked source, rebuilt, run, and reverted,
with the tree verified byte-for-byte afterwards (26 of 26 restored). **The
compiler exit status is printed beside each row**, because a mutation that fails
to build reads as a passing test, and this brick paid that toll once — see M13
below. `cases`/`assertions` are what `doctest` reported FAILING.

| id | mutation | compiler exit | result | cases | assertions | first failing case |
|---|---|---:|---|---:|---:|---|
| R0 | RED-FIRST: every W4b-1 mechanism neutralised at once | 0 | RED | 5 | 68 | dots3-note W4b-1: the sliding layer agrees with the independent reference |
| M1 | the WINDOW bound is dropped — the mask keeps causality only | 0 | RED | 4 | 27 | dots3-note W4b-1: the sliding layer agrees with the independent reference |
| M2 | the window bound rewritten as `> q - W` (an INTEGER-EQUIVALENT form) | 0 | **GREEN** | 0 | 0 | — |
| M2b | the window bound is OFF BY ONE — it admits W + 1 keys | 0 | RED | 4 | 12 | dots3-note W4b-1: the sliding layer agrees with the independent reference |
| M3 | the CAUSAL half of the mask is dropped | 0 | RED | 4 | 10 | dots3-note W4b-1: the sliding layer agrees with the independent reference |
| M4 | the mask ignores `valid` — unmapped and past-the-end slots score | 0 | RED | 1 | 1 | dots3-note W4b-1: the score mask is a POSITION predicate, not a token-order one |
| M5 | the query position is the BATCH INDEX, not the absolute position | 0 | RED | 1 | 23 | dots3-note W4b-1: the score mask is a POSITION predicate, not a token-order one |
| M6 | the gather reads at the LOGICAL stride over a PADDED cache | 0 | RED | 1 | 16 | dots3-note W4b-1: the KV gather reads PAGED rows at the PHYSICAL stride |
| M7 | the gather ignores the BLOCK TABLE and reads pages flat | 0 | RED | 4 | 27 | dots3-note W4b-1: the sliding layer agrees with the independent reference |
| M8 | an UNMAPPED page is reported VALID | 0 | RED | 1 | 3 | dots3-note W4b-1: the KV gather reads PAGED rows at the PHYSICAL stride |
| M9 | the gather starts at 0 instead of `max(seq_len - gather_len, 0)` | 0 | RED | 1 | 10 | dots3-note W4b-1: the KV gather reads PAGED rows at the PHYSICAL stride |
| M10 | `gather_len` is not rounded up to the kernel's BLOCK_T | 0 | RED | 4 | 428 | dots3-note W4b-1: the sliding layer agrees with the independent reference |
| M11 | the metadata gathers the WHOLE context instead of the window union | 0 | RED | 1 | 0 | dots3-note W4b-1: the windowed metadata caps every gather at the window and packs chunks that fit |
| M12 | the chunk plan ignores the workspace budget | 0 | RED | 1 | 1 | dots3-note W4b-1: the windowed metadata caps every gather at the window and packs chunks that fit |
| M13 | `_logical_cache` narrows at the LOGICAL stride | 0 | RED | 1 | 33 | dots3-note W4b-1: `_logical_cache` narrows a PADDED row, and the logical stride does not |
| M14 | the cache write runs to the PHYSICAL row and treads on the padding | 0 | RED | 1 | 10 | dots3-note W4b-1: `_logical_cache` narrows a PADDED row, and the logical stride does not |
| M15 | the masked score becomes -inf instead of upstream's -FLT_MAX literal | 0 | RED | 1 | 43 | dots3-note W4b-1: the score mask is a POSITION predicate, not a token-order one |
| M16 | the sliding arm inherits the MODEL-LEVEL rope theta | 0 | RED | 2 | 3 | dots3-note W4b-1: the SLIDING geometry comes off the RELEASED config, and it is NOT the full one |
| M17 | the softmax scale uses the LATENT row instead of `qk_head_dim` | 0 | RED | 4 | 7 | dots3-note W4b-1: the SLIDING geometry comes off the RELEASED config, and it is NOT the full one |
| M18 | the window is resolved one wider than `sliding_window_size` | 0 | RED | 3 | 7 | dots3-note W4b-1: the SLIDING geometry comes off the RELEASED config, and it is NOT the full one |
| M19 | the absorbed query takes the NoPE lanes as its rope half | 0 | RED | 3 | 6 | dots3-note W4b-1: the sliding layer agrees with the independent reference |
| M20 | the sliding geometry accepts a params object that claims an indexer | 0 | RED | 1 | 1 | dots3-note W4b-1: the sliding geometry refuses what is not it |
| M21 | the layer gathers a DECODE-sized window for a whole-prefill batch | 0 | RED | 1 | 2 | dots3-note W4b-1: the sliding layer agrees with the independent reference |
| M22 | REACHABILITY: the cache row stops coming from the resolved params | 0 | **GREEN** | 0 | 0 | — |
| M23 | REACHABILITY: the two LoRA scales stop being resolved from the config | 0 | RED | 2 | 7 | dots3-note W4b-1: the SLIDING geometry comes off the RELEASED config, and it is NOT the full one |
| M24 | REACHABILITY: the head count comes from the FULL arm's params | 0 | RED | 2 | 3 | dots3-note W4b-1: the SLIDING geometry comes off the RELEASED config, and it is NOT the full one |

**THE CROSS-CHECK, through the tool this tree already ships.** Every row above
that `scripts/mutation-harness.py` can express — the eight single-substitution
rows spanning all four §2.3 mechanisms plus the two reachability rows — was run
through it a second time, on a clean tree, and it agreed with the scratch driver
on every count:

```sh
python3 scripts/mutation-harness.py --build build-w4b \
    --test test_dots3_note_attn --plan $SCRATCH/w4b/xcheck.jsonl
```

```
BASELINE test_dots3_note_attn: exit=0 cases=30 (0 failed) assertions=2418 (0 failed)
M1-window-bound-dropped            BUILT: YES  cc-err 0  EXIT 1  30/4F  2400/27F  DETECTED
M2b-window-off-by-one              BUILT: YES  cc-err 0  EXIT 1  30/4F  2400/12F  DETECTED
M6-gather-logical-stride           BUILT: YES  cc-err 0  EXIT 1  30/1F  2418/16F  DETECTED
M9-gather-start-zero               BUILT: YES  cc-err 0  EXIT 1  30/1F  2418/10F  DETECTED
M11-metadata-no-window-cap         BUILT: YES  cc-err 0  EXIT 1  30/1F  2392/0F   DETECTED
M14-cache-write-treads-on-padding  BUILT: YES  cc-err 0  EXIT 1  30/1F  2418/10F  DETECTED
M17-scale-uses-latent-row          BUILT: YES  cc-err 0  EXIT 1  30/4F  2400/7F   DETECTED
M24-head-count-from-full-arm       BUILT: YES  cc-err 0  EXIT 1  30/2F  2418/3F   DETECTED
```

Eight for eight, same failing case counts and same failing assertion counts as
the table above. That is worth more than a tidier provenance line: two
independently written drivers, one of them tracked in this repository, produce
the same verdict on the same defects. `M11`'s `0F` is a case that THREW rather
than asserted, which is why the harness reads the EXIT CODE and not the summary
line — its own docstring names that trap.


**R0 is the RED-first arm and it ran BEFORE any green result was recorded.**
With the window mask a no-op, the gather reading at the logical stride, and both
LoRA rescales and `k_rope_only_layernorm` dropped from the sliding forward, the
gate reads 5 cases / 68 assertions failing, exit 1, compiler exit 0. That is a
real run and not an inference.

**M13 FAILED TO BUILD on its first attempt and that is recorded, not tidied
away.** Rewriting `_logical_cache`'s inner loop to read at the logical stride
leaves the `src` pointer unused, and `-Werror=unused-variable` stops the build —
compiler exit 1, `NOT A RESULT`. The driver prints the exit beside every row for
exactly this reason (spec §4.6's stale-binary finding), and it also refuses a
binary older than the build, so the previous binary's green could not be read as
the mutation's. Re-run with `(void)src;` the row compiles clean and takes the
gate RED on 1 case / 33 assertions. **One row of 26, and it is the row that
would have read as a passing test.**

**TWO mutations came back GREEN, and neither is a hole.**

**M2 is a green the DRIVER earned, not the code.** `kv_pos > query_pos - W` and
`kv_pos >= query_pos - W + 1` are the SAME predicate over integers, so the row
measures the mutation and not the guard. It is kept in the table rather than
deleted, because a reader who sees "off-by-one, GREEN" and stops has learnt the
wrong thing; **M2b** is the real off-by-one — `>= query_pos - W`, which admits
`W + 1` keys — and it takes the gate red on 4 cases / 12 assertions. The window
bound is additionally pinned from both sides by the layer case: at `W == T` the
windowed answer equals the dense causal one to 3.9e-16, and at `W == T - 1`
exactly one query loses exactly one key and the layer moves by 0.0836.

**M22 is a green that is DEFINITIONALLY unreachable.** It stops
`ForwardSlidingAttention` reading `dims.physical_latent_row` and hard-codes the
layer's own `latent_row()` — and for the SLIDING arm those are the same number
by construction (`physical_head_size = swa_kv_lora_rank + swa_qk_rope_head_dim`,
model.py:283), because the padding exists for the FULL layers. So there is no
input that distinguishes them on this path. The reachability claim it was meant
to make is made by **M23** and **M24** instead, which stop the two LoRA scales
and the head count coming from the config and both go red; the padded row's own
reachability is carried by the device-refusal case, which reads
`p.physical_latent_row()` off the parsed params. Recorded rather than dropped,
because "the field is read" and "a mutation can see it being read" are different
statements and this row can only make the first one.

#### Two fixture defects a green mutation found, and both were FIXED

Spec §4.6's F1 said the answer to a mutation slipping under a bound is to sharpen
the FIXTURE, not to widen the bound. Two mutations here found the same disease
one field over — a fixture in which a real defect is a numeric NO-OP — and both
were answered by changing the geometry.

- **`gather_start` was never exercised.** Pinning `gather_start = 0` instead of
  `max(seq_len - gather_len, 0)` came back GREEN against every case, because in
  a PREFILL `gather_len >= window + T - 1 >= T` and the maximum IS the identity.
  Only a DECODE — one query at the tail of a long context — makes it fire. The
  gather case grew a decode-shaped arm (20 cached tokens over seven SHUFFLED
  pages, `gather_len` 8, so `gather_start` is 12) with the expected physical
  slot derived by hand per token, and **M9** now reds on 1 case / 10 assertions.
  This matters beyond the row: skipping the head of the context is precisely
  what makes a 513-wide window affordable on a 524288-position model.
- **`qk_head_dim` equalled `latent_row`.** At `swa_qk_nope = 6` the sliding
  `qk_head_dim` (6+4) and its `latent_row` (6+4) coincided, so **M17** —
  scaling by the latent row instead of `qk_head_dim`, a genuinely plausible
  confusion, since the absorbed MQA dots over 1088 lanes and scales by
  256^-0.5 — reddened only the released-config assertion and left the layer
  untouched. `swa_qk_nope` is now 8, and M17 reds on 4 cases / 7 assertions.
  The same disease sat one field over: `swa_num_attention_heads` equalled the
  full arm's; it is now 3 against 2, mirroring upstream's 64 against 128.
  **The reason W4b-1 first gave for that change was WRONG, and the fresh review
  measured it.** W4b-1 credited the separation with making **M24** detectable.
  It does not: applying M24 *and* setting `swa_heads` back to 2 in the same arm
  still takes the gate RED, on the RELEASED-config case, because the bench
  builds its weights from the RESOLVED dims — so a wrong head count builds a
  consistently wrong bench that the layer comparison cannot see, and the
  released-config assertions are what catch it. The separation is still worth
  having, for the reason below, but not for the reason first written.

#### One thing the gate cannot say, measured rather than assumed

**`valid` is redundant with causality everywhere except an unmapped page**, and
that is a property of the upstream predicate rather than a weakness here.
`logical_valid` excludes slots past `seq_len` (attention.py:79), but those slots
carry `kv_pos > query_pos` and the mask's causal half already drops them. So
`page_valid` earns its keep on exactly one input: a NEGATIVE block-table entry.
**M4** — the mask ignoring `valid` — therefore reds on exactly ONE assertion,
the one that holes a mapped slot inside the window, and **M8** — an unmapped
page reported valid — reds on three. Both numbers are small on purpose and the
reason is stated here rather than being read as thin coverage.

#### No regression on the sibling gates

Re-run at this head, all four unmoved from the numbers §4.6 recorded:
`test_dots3_note_scaffold` **26 cases / 110818 assertions / 0 failed**;
`test_mla_attention_block` **12 / 2247715 / 0**; `test_deepseek_v2_forward`
**11 / 1052 / 0**; `test_deepseek_v2_decode_graph_seam` **3 / 230 / 0**. No seam
file was touched, so no byte-identity probe was owed and none was run — W4a's
six-arm fingerprint stands as the seam's evidence and W4b-1 adds nothing to it.

### 4.8 W4b-2 put the SLIDING arm on the decode path, over a PADDED cache

**LANDED at W4b-2** (`row/MODEL-MM-dots3-note-W4b-2`; `include/vt/ops.h` +
`src/vt/ops.cpp` + `src/vt/cpu/cpu_mla_attn.cpp` + `src/vt/cpu/cpu_mla_prefill.cpp`
+ `src/vt/cuda/cuda_mla_attn.cu` + `src/vt/cuda/cuda_flash_attn_fa2.cu` (the two
windowed kernels), `include/vllm/model_executor/models/mla_attention.h` +
`src/vllm/model_executor/layers/attention/mla_attention.cpp` +
`include/vllm/model_executor/layers/attention/mla_chunked_context.h` +
`include/vllm/v1/attention/backend.h` + `src/vllm/v1/attention/backend.cpp`
(the seam), `src/vllm/model_executor/models/dots3_note.h` +
`dots3_note_device.cpp` (the wiring), and four test files). CPU-only when it
landed, because W4b-2 took no GPU lease. **That CUDA debt is now PAID, and NOT
by the host §6.3 designates.** An `rc run` lease on `orin:gpu0` (sm_87) compiled
and executed both CUDA halves on 2026-08-26. Thor could never have served the
prefill half, because its sm_110 is outside the `fa2` feature row. The evidence
is below rather than implied.

**Upstream re-derived at vLLM `origin/main` = `bc2d63e650`.**
`git diff --stat d9fbe526c0 origin/main -- vllm/models/dots3_note/` is EMPTY, so
`vllm/models/dots3_note/` is byte-identical to the tree W4b-1 read and every
§2.3 anchor holds unchanged at the newer head. The anchors this brick leans on,
each re-derived rather than copied: `_gather_swa_kv_kernel:49`,
`_apply_swa_score_mask_kernel:119` (the two mask predicates at `:151` and
`:152`), `_build_sliding_window_metadata:192`, `Dots3NoteFlashAttnPrefillBackend
:258` with `run_sliding_window:279` and its `window_size=(sliding_window - 1, 0)`
at **`:300`**, `Dots3NoteTritonMLAImpl:439` (the subclass that keeps
`self.sliding_window`, `:454-468`) with `_forward_swa_mqa:470` and
`forward_mha:565`, `Dots3NotePaddedSparseImpl:697` with `_logical_cache:700-702`
and `do_kv_cache_update:704-720`, `Dots3NoteSlidingAttention:329` (its scale
`qk_head_dim**-0.5` at `:446`, its rope at `:401-409`, `sliding_window=
config.sliding_window_size` at `:457`, `self.indexer = None` / `is_sparse =
False` at `:432-434`), `Dots3NotePaddedMLAAttention:204-216` and the
`physical_head_size` it is fed at `:283`, and the layer dispatch at `:501-505`.

#### The window is a KERNEL BOUND here and a GATHER upstream, and that is the same function

Upstream's decode does not window the paged kernel. `_forward_swa_mqa` gathers
`[max(seq_len - GATHER_LEN, 0), ...)` into a workspace, where `GATHER_LEN` is
`sliding_window + query_len - 1` **rounded up to 8** (`:484`), and then masks the
scores with `kv_positions <= query_position` and
`kv_positions >= query_position - WINDOW_SIZE + 1` (`:151-152`). The gather is a
SUPERSET and the mask is what makes it exact; the round-up exists because Triton
needs a power-of-two tile.

Walking the paged block table directly over `[seq_len - W, seq_len)` reaches the
identical key set with no gather, no workspace and no mask. So the port is a
`window_size` on the two MLA ops rather than two new ops, and the shared seam is
extended rather than bypassed — which is also why W4b-1's
`GatherSwaKv` / `ApplySwaScoreMask` / `BuildSlidingWindowMetadata` stay HOST
reference code driving the gate's oracle, exactly as W3's `ForwardFullAttention`
did after W4a. That is recorded here rather than left for a reader to infer, and
it is listed under `## Owed`.

`std::nullopt` is the ABSENT state on both args structs, and it is a NOT-TAKEN
branch rather than a wide window:

| where | what absence does |
|---|---|
| `cpu_mla_attn.cpp` | `j_start` stays 0 — the same `for (j = 0; j < seq_len; ++j)` loop the op had |
| `cuda_mla_attn.cu` | `kv_start` stays 0 in BOTH split stages, so the partition over `[0, seq_len)` is the one that was there |
| `cuda_flash_attn_fa2.cu` (MLA prefill) | `is_local` is false, so `is_causal` / `window_size_left` / `window_size_right` take their previous assignments and the template dispatch is unchanged |
| `cpu_mla_prefill.cpp` | the `first` lower bound stays 0 |
| `mla::MlaBlockDims::sliding_window` | 0, so no `window_size` is ever constructed |
| `v1::TritonMLAImpl::sliding_window` | 0, same |

**The op gates prove that rather than assert it.** On both ops a window at least
as wide as the longest sequence is compared against no window **bit-for-bit**,
not to a tolerance. A mask applied after the fact could not pass that.

#### The padded row needed ZERO `vt` changes, and the narrowing is one line

`Tensor::Slice(2, 0, logical)` shrinks `shape[2]` and keeps both leading strides
(`tensor.cpp:70-84`), every MLA cache op sources its strides from the tensor, and
that IS upstream's `kv_cache[..., : self.head_size]`. The narrowing is one line
in `Dots3NoteModel::ForwardDevice`. This is the correction §4.7 already recorded,
executed: the physical row is the 1088 both classes share, a FULL layer reads its
logical 576 out of the head of it, and a SLIDING layer's logical row IS the
physical one by construction, so the slice is the identity there and is written
unconditionally rather than branched.

**The evidence is the RAW cache bytes after a real forward**, not an argument:
the gate reads the pool back and asserts that lanes `[6, 10)` of every slot a
FULL layer wrote are still ZERO, with the CONTROL that the same lanes on the
sliding layers carry **28** non-zero values. Without the control the assertion
would pass on a fixture that produced zeros anyway.

#### Two of W4a's three refusals are LIFTED; the third is NARROWED; one is NEW

| refusal | at W4a | at W4b-2 |
|---|---|---|
| any `sliding_attention` layer | config level | **LIFTED** — runs through the same seam over `Dots3NoteSlidingAttnMlaDims` |
| a PADDED physical latent row | config level | **LIFTED** — `Slice(2, 0, logical)`, no `vt` change |
| a KV cache row disagreeing with the config | per step | **KEPT**, and now compared against the PHYSICAL row, which is what `MakeDots3NoteKVCache` tells the allocator |
| `seq_len > index_topk` | per step, always | **KEPT and NARROWED** — asked only of a config that HAS a full layer, because `Dots3NoteSlidingAttention` sets `self.indexer = None` / `is_sparse = False` (`model.py:432-434`) |
| a windowed prefill with chunked CONTEXT | — | **NEW**, in the seam |
| any MoE layer | config level | unchanged — W5 |
| a nextn tail | config level | unchanged — W10 |

**The new refusal is a scope statement with an upstream reason.** A sliding
layer's prefill gathers only `min(seq_len, query_len + W - 1)` keys and runs ONE
varlen call per request group (`attention.py:206`, `:594-654`); upstream never
merges context chunks under a window, so `forward_mha`'s LSE merge has no
windowed form to mirror. The seam throws rather than merging an UNwindowed
context into a windowed suffix, which is a silently wrong answer. Owed to W4b-3.

**The released `dots-studio/dots3-note-prev` config still refuses**, now at layer
1's MoE rather than at layer 2's sliding attention, so nothing a user can run
changed. `test_dots3_note_scaffold`'s forward-refusal case was updated to name
the piece the released config ACTUALLY trips on — a string that outlives the
refusal it describes is the failure this row keeps recording.

#### The gate, met

`test_dots3_note_attn` — **36 cases / 3028 assertions**, CPU-only, no GPU, no
checkpoint, no speed claim (30/2418 at W4b-1, 18/638 at W4a, 12/198 at W3).
Six new cases.

The bench is a MIXED config — `layer_types` `{full, sliding, full}`, every layer
a dense MLP, physical row 10 against the full arm's logical 6 — loaded through
`ModelRegistry::Resolve` → `reg.factory->load_weights` and run through
`ModelRegistry::Forward` **twice against one cache pool**: a six-token PREFILL,
then a DECODE of the seventh, over a SHUFFLED block table `{1, 0}`. Both halves
are compared against a whole-model double reference that dispatches per layer
kind into W3's `ref::Forward` and W4b-1's `sref::Forward` — a materialized MHA
with no cache, no paging, no gather and the window as the direct positional
predicate `s <= t && t - s < W`.

Running two steps against one pool is the point. The decode step reads K/V the
PREFILL step wrote, through the padded physical row and the shuffled table, so
"what the decode read out of the cache" has to equal "what a fresh full-sequence
forward computes".

**What the instrument measured, printed by the gate rather than assumed:** the
prefill's window cuts **3 of 6** queries and drops **6** keys; the decode query
at position 6 keeps **3** of its 7. At `window >= tokens` the windowed answer IS
the causal answer and every assertion here would pass on a port with no window
at all, which is why both counts are asserted BY NUMBER.

**The bound is `6e-2` and the three ratios are kept SEPARATE**, because §4.6's
review finding F1 is that merging them overstates the headroom:

| ratio | value | what it says |
|---|---:|---|
| bound / residue | 0.06 / 0.0254 = **2.36x** | headroom above the bf16 floor |
| nearest mechanism / bound | 0.158 / 0.06 = **2.63x** | headroom below the nearest defect |
| nearest mechanism / residue | 0.158 / 0.0254 = **6.22x** | separation of the whole instrument — a statement about the FIXTURE |

**The fixture was RETUNED twice, and both times a measurement forced it.** The
first draft ran four layers `{full, sliding, full, sliding}` with
`swa_rope_theta` 41 against 137 and amplified the sliding arm's k_pe rows 6x. It
measured a residue of **0.119** with the nearest mechanism — the sliding arm
inheriting the model-level rope theta — at **0.106**, i.e. the nearest defect
sat UNDER the quantisation floor and the instrument could not see it. Two
changes fixed it, and neither was widening the bound: the thetas became 3
against 1300, orders apart the way the released 5e4 against 8e7 is (W4b-1's
0.0300 relative on ONE layer is simply too small to survive a bf16 model), and
the schedule dropped to three layers, which is still full/sliding/full so a
per-layer field leaking in EITHER direction is wrong. Residue 0.119 → 0.0254,
nearest mechanism 0.106 → 0.158.

**Each sliding-only mechanism is shown EXERCISED, not merely compiled**, by
neutralising it in the REFERENCE and measuring the device arm drifting away.
Both ratios are given and labelled, for the reason above:

| the reference with … | device-vs-reference | / the 6e-2 BOUND | / the 0.0254 RESIDUE |
|---|---:|---:|---:|
| **no window at all** — plain causal attention | 0.818662 | 13.6x | 32.2x |
| the sliding arm inheriting the MODEL-level rope theta | 0.182502 | 3.04x | 7.18x |
| the sliding arm's **q** LoRA rescale dropped | 0.158023 | 2.63x | 6.22x |
| the sliding arm's **kv** LoRA rescale dropped | 1.10842 | 18.5x | 43.6x |
| the sliding arm's `k_rope_only_layernorm` dropped | 0.672635 | 11.2x | 26.5x |
| the sliding arm's headwise gate made lane-wise | 0.200210 | 3.34x | 7.88x |

The two LoRA scales are neutralised SEPARATELY as well as being different
numbers on this fixture (`sqrt(16/3)` and `sqrt(16/6)`), so an arm that dropped
both at once could not distinguish a port carrying both from one carrying only
the q. Upstream's released ranks make the two SLIDING scales EQUAL at
`sqrt(5120/1024)`; the fixture deliberately does not copy them, and the
released-config case pins the released values separately.

#### The op gates, and why their oracle is the op itself

`vt::MlaDecodeAttention` and `vt::MlaPrefillAttention` are gated WITHOUT writing
a windowed reference, deliberately: a reference that recomputed
`seq_len - 1 - left` or `iq + (lk - lq) - left` a second time would share the
arithmetic it is supposed to check, which is the shared-helper trap this project
keeps naming.

- **Decode.** The windowed call over a length-`n` PAGED sequence is compared
  against an UNWINDOWED call over a freshly built single-page cache holding
  exactly that request's last `min(W, n)` keys — a path already gated against
  the ported `ref_mla` oracle. The window is **13** against pages of **16**, so
  its start lands INSIDE a page and a port that rounded to a page boundary is
  caught. The boundary is pinned from BOTH sides: a window one key WIDER is a
  different answer on every request the window cut.
- **Prefill.** The windowed multi-query call is compared against an EXPANDED
  batch in which every query becomes its own single-query request carrying only
  the keys its window admits, run UNWINDOWED. With `lq == 1` the bottom-right
  causal bound admits every key handed in, so the expansion needs no mask of its
  own. **475** (query, key) pairs are dropped across 57 queries.
- Both ops refuse a `right != 0` window BY NAME, and the prefill additionally
  refuses a NON-causal one: FlashAttention's local mask REPLACES the causal
  specialization (`is_causal = causal && !is_local`), so "everything forward,
  windowed backward" has no finite spelling. Upstream never asks for one.

`test_ops_mla_attn` **15 cases / 246290 assertions** (11 / 197113 at the base
SHA `925a4a587`); `test_ops_mla_prefill` **6 cases / 329772 assertions**
(4 / 242156 at the base). Both base numbers are MEASURED — the two test files
were checked out at the base SHA, rebuilt and run, then restored and verified
byte-for-byte — rather than counted off `TEST_CASE` lines.

#### The DeepSeek-V2 path is byte-identical, MEASURED again on six arms

`mla::ForwardMlaAttentionBlock` still has FOUR callers — `deepseek_v2`,
`minicpm3`, `kimi_linear` and `dots3_note` — and DeepSeek-V2-Lite carries a
SACRED token-exact gate that cannot be run on a box with no GPU and no
V2-Lite checkpoint. W4a's standard applies unchanged, and the probe was rebuilt
rather than reused.

The BEFORE arm is a separate `git archive` tree at the base SHA `925a4a587`,
with a byte-identical probe appended (`md5sum` equal on both files), its own
`cmake` configure and its own build — so there is no previous binary a failed
compile could fall back to. Both arms print the compiler exit and refuse a binary
older than its source.

| arm | geometry | bytes | BASE `925a4a587` | HEAD |
|---|---|---:|---|---|
| 0 | V2-Lite, f32, MIXED (2 decode + 2 prefill, one with context) | 106496 | `2071435139082975929` | identical |
| 1 | V2-Lite, bf16, same batch | 53248 | `15607516550467795365` | identical |
| 2 | V3 q_lora branch, f32 | 86016 | `5937425064452249605` | identical |
| 3 | V3 q_lora branch, bf16 | 43008 | `4610065661939359460` | identical |
| 4 | MiniCPM3 (`is_neox_style=true`), f32 | 30720 | `7108812291202172077` | identical |
| 5 | MiniCPM3 (`is_neox_style=true`), bf16 | 15360 | `16826999257951116139` | identical |

Six for six.

**Arms 0 and 1 reproduce W4a's recorded fingerprints exactly — and arms 2 to 5
do NOT. The first draft of this paragraph generalised from the first two, and
the #1969 review caught it.** What is actually measured:

| arm | §4.6, base `d7d1ee914` | §4.8, base `925a4a587` | |
|---|---|---|---|
| 0 | `2071435139082975929` | `2071435139082975929` | same |
| 1 | `15607516550467795365` | `15607516550467795365` | same |
| 2 | `4982522374592074643` | `5937425064452249605` | DIFFERENT |
| 3 | `3757253798370478450` | `4610065661939359460` | DIFFERENT |
| 4 | `9024916185557934982` | `7108812291202172077` | DIFFERENT |
| 5 | `16077001697345918067` | `16826999257951116139` | DIFFERENT |

**It is a PROBE difference and not a behaviour change, and that conclusion is
measured rather than argued.** Three checks, each cheap and each re-run at this
head:

1. `git log --oneline d7d1ee914..925a4a587` over `mla_attention.{h,cpp}`,
   `cpu_mla_attn.cpp`, `cpu_mla_prefill.cpp`, `deepseek_v2.cpp` and
   `minicpm3.cpp` returns EXACTLY ONE commit — `446ac1806`, which is W4a itself
   — and W4a's own gate is the §4.6 table asserting byte-identity across it.
   Widening the sweep to all of `src/vt` and `include/vt` over the same range
   adds only `Conv3d` and `Exl3Gemm`, neither of which the MLA block reaches.
   So base-to-base the executing chain is unchanged, and by transitivity all
   four measurements describe one function.
2. `grep -r 2071435139082975929` over the tree hits ONLY this file. The probe
   was never committed — §4.6 says so in its own words — so the two tables were
   produced by two independently hand-written instruments in two sessions.
3. Arms 0 and 1 are V2-Lite, whose dims both authors would write the same way
   and which take no `q_lora_scale` and no `is_neox_style`. Arms 2 to 5 carry
   free parameters that each author chose, and `q_lora_scale` in particular did
   not EXIST at §4.6's base: `git show d7d1ee914:…/mla_attention.h | grep -c
   q_lora_scale` is **0** and the same grep at `925a4a587` is **2**, because
   446ac1806 introduced the field. §4.6's probe had to be byte-identical across
   its own two trees, so it could not mention the field at all; §4.8's probe
   could. The byte COUNTS agree on all six arms (86016, 43008, 30720, 15360),
   so the two probes agreed on shapes and differed on values — which is exactly
   the signature of a differing scalar or weight fill.

**The transferable rule, which is why this stays in the record rather than
being quietly corrected: a fingerprint from an uncommitted, hand-written probe
is not a cross-session reproducible quantity.** Two probes sharing a prose
label — "the V3 q_lora arm" — are two different instruments, and comparing
their outputs measures the authors, not the code. What each table legitimately
says is base-vs-head identity WITHIN its own session, which is the claim each
was built to make. The cross-table agreement on arms 0 and 1 is a pleasant
coincidence of a parameter-free fixture, not evidence of reproducibility.

Committing the probe is what would fix this, and it is NOT done here: neither
scratch tree survives, and writing a third probe would produce a third set of
numbers and no more reproducibility than two. It is listed under `## Owed`.

**The probe's own false-green, caught by the harness rather than by luck.** The
first BASE run used doctest's `-ts=` (the test-SUITE filter) instead of
`--test-case=`. It matched ZERO cases, printed `[doctest] test cases: 0 | 0
passed | 0 failed | 13 skipped` and `Status: SUCCESS!`, and exited 0. Read
without checking the case count that is a clean pass with no fingerprints — the
third of the four failure modes `scripts/mutation-harness.py`'s own docstring
enumerates, met in the one place that was hand-driven rather than run through the
harness. The rule generalises: **a filter that matches nothing is not a result,
and only a NON-ZERO case count says so.**

**Both DeepSeek gates were re-run at this head.** `test_mla_attention_block`
**12 cases / 2247718 assertions** and `test_deepseek_v2_forward` **11 / 1052**.
The CASE counts are unmoved from the numbers §4.6 recorded, because the #1969
review repair added a SUB-BLOCK and not a thirteenth case. The assertion count
is NOT unmoved: that repair added the `(b2)` block for the
`MlaBlockDims::sliding_window < 0` refusal INSIDE `TEST_CASE` #12, "MLA block:
the dots3-note fields REFUSE what they cannot represent, by name"
(`tests/vllm/model_executor/layers/attention/test_mla_attention_block.cpp:1192-1204`,
with its two boundary controls immediately after at `:1205-1214`). It is worth
+3 assertions over §4.6's 2247715, and the R1 mutation row below is that block's
red-proof. The implementer, a fresh reviewer and the operator each measured
2247718 independently at the W4b-2 head and agree. §4.6's and §4.7's 2247715
stay as written, because 2247715 was the true value when each was measured.
`test_deepseek_v2_decode_graph_seam` **3 / 230** and `test_ops_mla_cache`
**9 / 2947** are unmoved. `test_dots3_note_scaffold` reads **26 / 110819** — one
assertion more than §4.6's 110818, because its forward-refusal case's single
`Contains("sliding-window MLA")` became two, `Contains("MoE layer")` and
`Contains("W5")`, naming the piece the released config now actually trips on.

**NOT run, and named rather than implied:** the SACRED DeepSeek-V2-Lite e2e token
gate. It needs a ~29.26 GiB checkpoint on a CUDA host; this brick ran CPU-only on
a box with neither.

#### The mutation table

**The driver is the one this tree ships.** Every row below was produced by
`scripts/mutation-harness.py` (row LTX25-RES2S-LOOP,
[#921](https://github.com/mudler/vllm.cpp/issues/921)) rather than by a scratch
script — which is what W4b-1's own section says it should have done. The harness
implements the four guards this project has paid for: it REFUSES an anchor that
does not occur exactly once, prints the COMPILER EXIT beside every row, runs the
whole binary and asserts a NON-ZERO case count, and re-stamps every restore so a
stale object cannot carry a previous binary forward. It also refuses to start on
a dirty tree, which is how a restore failure stays distinguishable from an edit.
A second guard runs before the first compile: `check_anchors.py` asserts every
anchor in the plan occurs exactly once, so a stale plan costs nothing.

```sh
python3 scripts/mutation-harness.py --build build-w4b2 \
    --test test_dots3_note_attn --plan $SCRATCH/w4b2_plan.jsonl
```

**Every count below was RE-MEASURED at the final baseline** (36 cases / 3028
assertions), not carried over from the first pass. The first pass ran at
35/3025, before the case M16 exposed as missing existed; a table that mixed the
two would be an instrument reporting on a state it was not given, which is the
failure this row keeps naming. The verdicts agreed across both passes.

| id | mutation | compiler exit | result | cases | assertions |
|---|---|---:|---|---:|---:|
| M1 | the sliding window is never resolved from the config | 0 | RED | 2 | 3 |
| M2 | the window reaches the ops ONE WIDER (`sliding_window`, not `- 1`) | 0 | RED | 1 | 1 |
| M3 | the CPU decode's window START is off by one (`seq_len - left`) | 0 | RED | 1 | 1 |
| M4 | the CPU decode ignores the window | **1** | **NOT A RESULT** | — | — |
| M4b | the same, with `(void)j_start` so it compiles | 0 | RED | 1 | 1 |
| M5 | the CPU prefill's PASS-1 loop ignores the window | 0 | **GREEN** | 0 | 0 |
| M5b | the CPU prefill's `first` bound is forced to 0 in ALL THREE passes | 0 | RED | 1 | 1 |
| M6 | the padded cache is read at the LOGICAL stride | 0 | RED | 1 | 24 |
| M7 | the `_logical_cache` narrowing is dropped — a full layer reads the physical row | 0 | RED | 3 | 0 (threw) |
| M8 | a SLIDING layer runs the FULL arm's `MlaBlockDims` | 0 | RED | 4 | 1 |
| M9 | one shared rope cache for both geometries | **1** | **NOT A RESULT** | — | — |
| M9b | the same, compiling | 0 | RED | 3 | 3 |
| M10 | the materializer uses the FULL dims for every layer | **1** | **NOT A RESULT** | — | — |
| M10b | the same, compiling | 0 | RED | 4 | 3 |
| M11 | REACHABILITY: the DECODE production call site is deleted | 0 | RED | 1 | 1 |
| M12 | REACHABILITY: the PREFILL production call site is deleted | 0 | RED | 1 | 1 |
| M13 | the per-step cache-row check is deleted | 0 | RED | 2 | 2 |
| M14 | the `index_topk` refusal is asked of EVERY config | **1** | **NOT A RESULT** | — | — |
| M14b | the same, compiling | 0 | RED | 1 | 1 |
| M15 | the `index_topk` refusal is deleted outright | **1** | **NOT A RESULT** | — | — |
| M15b | the same, compiling | 0 | RED | 2 | 2 |
| M16 | the windowed-prefill-with-CONTEXT refusal is deleted | 0 | **GREEN** | 0 | 0 |
| M16b | the same, after the missing case landed | 0 | RED | 1 | 1 |
| M17 | the sliding softmax scale uses the LATENT row, not `qk_head_dim` | 0 | RED | 1 | 1 |
| M18 | the window is ONE WIDER at the impl (`dims.sliding_window + 1`) | 0 | RED | 1 | 1 |
| M19 | M18, measured on `test_deepseek_v2_forward` | 0 | **GREEN** | 0 | 0 |
| M20 | M18, measured on `test_mla_attention_block` | 0 | RED | 3 | 4 |

**FIVE of twenty-seven rows FAILED TO BUILD on their first attempt, and every one
of them is printed rather than tidied away.** M4, M9, M10, M14 and M15 each leave
a variable unread (`j_start`, `rope_swa`, `sliding`, `has_full_layer`) and
`-Werror=unused-variable` stops the build — compiler exit 1, `NOT A RESULT`.
That is the failure mode a hand-driven pass reads as a passing test, and the
proportion is the point: **nearly one row in five** would have been a false
green here. W4b-1 hit it once in twenty-six; the harness caught it five times in
one plan without anyone looking. The `b` rows re-run the identical defect behind
`((void)x, …)`.

**M11 and M12 are the reachability rows, and they are the ones that say the
window is REACHED.** M11 stops `impl.sliding_window` being assigned from
`dims.sliding_window`, so no window ever reaches `vt::MlaDecodeAttention`; M12
passes a literal 0 to `ForwardMlaPrefillMha`, so none reaches
`vt::MlaPrefillAttention`. Both go red, so the window on the decode path comes
from the config through the real loader and the shared seam, not from a struct
the test typed.

**THREE GREEN ROWS, and each says something different.**

**M5 is a green the DRIVER earned, not the code — and it is the reason M5b
exists.** Forcing the PASS-1 loop to start at 0 makes the kernel compute logits
for out-of-window keys and take the running MAX over them. It changes nothing:
softmax is invariant to the constant subtracted before `exp`, and passes 2 and 3
still sum only `[first, visible)`. The only observable is the LSE, which this
config never merges because it has no chunked context. So the row measures the
mutation, not the guard. M5b moves the `first` DEFINITION instead, which reaches
all three passes with one substitution, and reds. Kept in the table rather than
deleted, because a reader who sees "prefill window ignored, GREEN" and stops has
learnt the wrong thing.

**M16 is a green that found a REAL GATE GAP, and it is the most valuable row
here.** Deleting the windowed-prefill-with-context refusal left the gate
completely green — because the case asserting that refusal never made it out of
the draft into the committed file. A refusal whose test does not exist is
indistinguishable from a refusal that works. The case is in the gate now, with
two controls (no window ⇒ the call proceeds and fails with a DIFFERENT exception
type; a window with no chunk list ⇒ it does not fire either), and M16b reds.

**M19 is a green that MAPS THE GATES rather than exposing a hole**, and its pair
M20 is why. Both apply the same defect — a window leaking onto the DeepSeek path
at `impl.sliding_window`. `test_deepseek_v2_forward` does not see it;
`test_mla_attention_block`, whose cases include decode-only and MIXED batches,
reds on 3 cases / 4 assertions. Together they say the field IS on the DeepSeek
path and its 0 IS load-bearing — measured, which is what the byte-identity table
one section up needs and cannot supply on its own, since identical output could
also mean nothing ever read the field.

**The MECHANISM this section first gave for M19's green was wrong, and the
#1969 review measured it wrong.** The claim was that `test_deepseek_v2_forward`
misses the defect "because its CPU synthetic forward drives the PREFILL half and
`impl.sliding_window` only reaches the decode MQA" — which implies the prefill
half of the same leak WOULD be caught. It is not. Leaking the window into the
PREFILL call instead (`suffix_lse_t, dims.sliding_window + 1`) was run against
that binary at this head through `scripts/mutation-harness.py`: compiler exit
**0**, EXIT 0, **11 cases / 1052 assertions, zero failures — SURVIVED**. The
same mutation on `test_mla_attention_block` is compiler exit 0, EXIT 1, 4 cases
and 2 assertions failing — DETECTED. So the defect is real and detectable, and
the deepseek binary's blindness is a property of the binary.

**The real reason is that `test_deepseek_v2_forward`'s CPU cases have NO VALUE
ORACLE for the attention output at all.** `:443` asserts finiteness and
run-to-run determinism, `:464` asserts fusion-catalog ADOPT equals the hand-call
fallback, `:482` asserts a zero-routed MoE layer equals a dense one, `:511`
asserts the shared expert changes the logits, and `:537` repeats `:443` at the
real V2-Lite head dims. Every one of them compares the model against ITSELF
under another configuration, or against nothing. A defect that moves both sides
of such a comparison by the same amount is invisible whichever half of attention
it lands in. That both halves are also prefill-only (`RunTiny` builds
`PrefillMeta`, `:299`) is true and is a SECOND reason the decode leak
specifically is unreached — but it is not the reason the binary is blind, and
stating it as the reason draws the map wrong. The M19 conclusion survives; its
explanation did not.

#### The #1969 REVIEW-REPAIR rows, measured after the repair

Six more rows, all through `scripts/mutation-harness.py` on this head, all with
the compiler exit printed. They close the review's F5 and ground its F3 and F4.

| id | mutation | binary | compiler exit | result | cases | assertions |
|---|---|---|---:|---|---:|---:|
| R1 | the `MlaBlockDims::sliding_window < 0` refusal is DELETED | `test_mla_attention_block` | 0 | **RED** | 1 | 1 |
| R1-control | the same, on the ROW gate | `test_dots3_note_attn` | 0 | GREEN | 0 | 0 |
| R2 | the per-step `ld.head_size() <= physical_row` check is DELETED | `test_dots3_note_attn` | 0 | GREEN | 0 | 0 |
| R3 | the CONFIG-level `physical >= logical` refusal is DELETED | `test_dots3_note_scaffold` | 0 | **RED** | 1 | 1 |
| R4 | the PREFILL window leaks onto the DeepSeek path | `test_deepseek_v2_forward` | 0 | GREEN | 0 | 0 |
| R4b | the same | `test_mla_attention_block` | 0 | **RED** | 4 | 2 |

**R1 and R1-control are the before/after pair.** The refusal shipped with no
test, and deleting it leaves the ROW gate completely green — which is what the
review measured and what R1-control reproduces here on an unchanged binary. The
new case in `test_mla_attention_block` is what turns it red, and it carries
controls on both sides of the boundary (0 is ABSENT and legal, 513 is legal) so
it cannot pass on an implementation that refused every window.

**R2 stays GREEN and is not a hole, and R3 is the measurement that says so.**
The per-step `ld.head_size() <= physical_row` check is UNREACHABLE, not merely
untested: `physical_latent_row()` IS `swa.latent_row()` (`dots3_note.h:192`), so
on a sliding layer it is an identity, and on a full layer
`ParseDots3NoteParams` has already refused the violating config at load
(`dots3_note.cpp:389`). R3 deletes THAT check and `test_dots3_note_scaffold`
reds, so the closure is gated rather than assumed. The review's diagnosis — that
`Tensor::Slice` backstops it, so deleting it only downgrades the message —
understates the position: the backstop is not reached either, because no input
the loader accepts can violate the invariant. The check is kept as the
executable spelling of upstream's `assert physical_head_size >= self.head_size`
(`model.py:210`), the site says all of this in a comment, and `## Owed` carries
it as an untested assertion rather than as a gated refusal.

**R4 and R4b are what correct M19's mechanism**, and the correction is written
into the M19 paragraph above rather than only here.

#### The merge that built clean and threw, and what caught it

`origin/main` moved twice while this brick was in flight, and the second
integration is worth recording. It merged with no conflict, it COMPILED with no
error, and `test_dots3_note_attn` then went from 36/3028 green to **5 cases
throwing**:

```
vt: resident weight: EMPTY tensor has no host bytes to alias
    (host-alias arm, dtype f32, rank 0)   dense_attn_block.h:206
```

The incoming commit was [#1952](https://github.com/mudler/vllm.cpp/pull/1952)'s
review finding [#1953](https://github.com/mudler/vllm.cpp/issues/1953):
`ResidentWeight` now REFUSES an empty weight, because an empty one aliases a null
host pointer that no downstream op can detect — every op validates rank, shape,
dtype and device, and the shape comes from the CALLER rather than from the bytes.
That refusal is right, and W4b-2 was on the wrong side of it: it made BOTH rope
caches resident up front while `MaterializeDots3NoteDevice` deliberately leaves
the unused one empty (each is 64 MiB at the released 524288 positions). The fix
is one guard per cache, and the comment now says the guard is the CONTRACT rather
than an optimization.

**Nothing about this was visible to the merge.** The two branches touched
different files, `git merge` reported clean, and `cmake --build` exited 0 — the
"merge-tree CLEAN is not merge-tree BUILDS" note one step further along, where it
builds too and only the gate can see it. The only thing that caught it was
re-running the focused gate set AFTER the merge and BEFORE the push, which is the
sequence AGENTS.md asks for and the reason it asks.

#### The 33 sliding layers get a FULL-LENGTH KV spec, and upstream gives them a windowed one

A divergence this brick creates and does not close, recorded here because a
token gate structurally cannot see it and `porting.md` asks for the memory
format to be compared with the oracle explicitly.

Upstream's `MLAAttention.get_kv_cache_spec` branches on the window
(`vllm/model_executor/layers/attention/mla_attention.py:1215-1219` @
`bc2d63e650`):

```python
if self.sliding_window is not None:
    return SlidingWindowMLASpec(**common_kwargs, sliding_window=self.sliding_window)
return MLAAttentionSpec(**common_kwargs, non_causal_multi_token_decode=...)
```

and every sliding layer takes that branch, because
`Dots3NoteSlidingAttention.__init__` passes
`sliding_window=config.sliding_window_size` into `MLAAttention`
(`vllm/models/dots3_note/nvidia/model.py:457`). Ours emits ONE uniform
`v1::MLAAttentionSpec` for all 46 layers
(`src/vllm/model_executor/models/dots3_note.cpp`,
`MakeDots3NoteKVCache`).

**There is no correctness consequence** — the window is applied on READ, by the
two ops, and the gate above proves that it is. The consequence is ALLOCATION.
`SlidingWindowSpec.max_admission_blocks_per_request` caps a windowed layer at
`cdiv(min(sliding_window - 1 + extra_retained + in_flight, max_model_len),
block_size) + 1` blocks (`vllm/v1/kv_cache_interface.py:696-722`), against a
full-length spec's whole `max_model_len`. On the released config that is 513
against 524288, on 33 of 46 layers — 72% of the tower holding a full-length
latent cache where upstream holds roughly a window's worth. It is the single
largest memory property of this architecture, and no token gate can report it,
because the tokens are right either way.

**What is missing has SHRUNK from three pieces to one, and the remainder is
another row's work rather than this row's.** The earlier text here read "THREE
pieces are missing, not one" and named W4b-3 as the doer of all three. Both
halves of that are now wrong, and each piece is stated separately so a reader
can check it.

- **`SlidingWindowMLASpec`, the spec TYPE, has LANDED.** It was on the list
  `include/vllm/v1/kv_cache_interface.h` records as deliberately omitted at MLA
  campaign T1. KV-DSV4-MULTICACHE W1
  ([#1960](https://github.com/mudler/vllm.cpp/issues/1960), `c1e6f3fb9`) landed
  it on `main` while W4b-2 sat in review, so `struct SlidingWindowMLASpec :
  SlidingWindowSpec` is in that header today and any text calling it omitted
  from this tree is false. The header records the landed shape as ALLOCATION
  METADATA ONLY, with nothing outside tests constructing it.
- **`SlidingWindowSpec::max_admission_blocks_per_request`, the formula quoted
  just above, is PRESENT** (`include/vllm/v1/kv_cache_interface.h:358-361`). The
  tree can already state a windowed layer's admission bound.
- **`max_memory_usage_bytes` is the ONE piece still genuinely absent** from that
  header (`:64`), so the tree cannot yet express the SAVING as a number.
- **The heterogeneous per-layer KV-cache GROUP SPLIT is NOT this row's work.**
  It is `KV-DSV4-MULTICACHE` W3, "the runner carries more than one attention
  group and more than one cache per layer", which that spec calls the wave that
  touches every model, with W4 for non-uniform block sizes
  ([`kv-dsv4-multicache.md`](kv-dsv4-multicache.md), `## Work breakdown`). The
  runner today selects the FIRST non-eagle full-attention or MLA group as its
  target (`src/vllm/v1/worker/gpu/runner.cpp:703-712`). **It does NOT pass over
  the rest in silence.** Since KV-DSV4-MULTICACHE W3 (`ca3dcda21`) a leftover
  group puts the runner on the generalized multi-cache path (`:784-800`) and it
  ALLOCATES; the `VT_CHECK` at `:860-870` REFUSES only what that path cannot
  represent — a spec that is neither an `AttentionSpec` nor a `MambaSpec`, a
  SECOND recurrent group, an EAGLE draft group, and a group whose published
  layer names do not all resolve to distinct in-range indices — and it names the
  count, the kind, the first layer and the page size:
  "N published KV cache group(s) get NO cache from this runner ... Refusing
  rather than allocating a SUBSET of the published topology in silence", and it
  names `KV-DSV4-MULTICACHE` W3 as the owner of lifting the limit. That refusal
  landed at `6b18829bc` (KV-DSV4-MULTICACHE W2,
  [#1973](https://github.com/mudler/vllm.cpp/issues/1973)), which reached this
  branch through its own merge of `origin/main`. It is gated on the exact two
  shapes this row would publish, a `kSlidingWindowMla` group and a SECOND
  `kMlaAttention` group (`tests/vllm/v1/worker/test_runner.cpp:1621` and
  `:1643`). So emitting a second spec kind here before that wave lands makes the
  runner THROW at construction. The DEPENDENCY did not change. Only the failure
  mode did, from a silent short allocation to a named refusal. dots3-note
  DEPENDS on that row. It must not duplicate it.

The DIVERGENCE itself is unchanged: we still emit one uniform spec for all 46
layers. Owed, under `## Owed`, with the per-layer emission owned here and
BLOCKED ON `KV-DSV4-MULTICACHE` W3/W4
([#1925](https://github.com/mudler/vllm.cpp/issues/1925)).

#### The CUDA half is COMPILED and EXECUTED, on sm_87

Both CUDA changes are small and local: `kv_start` in the two MLA-decode split
stages, and the `is_local` normalization the paged FA-2 launcher already performs
one function above the MLA one. W4b-2 was written on a box with no GPU and no
`nvcc`, so for most of this brick's life this section read "written, not
compiled, not run". An `rc run` lease on **`orin:gpu0`** (Jetson AGX Orin, sm_87,
about 36 minutes of device time) closed both halves of that on 2026-08-26. The
SHA was PROVEN rather than asserted: the job cloned in the container and refused
to build unless `git rev-parse HEAD` equalled
`53424910dfa31fbd10bcb3296a12401eaed8ee54` with `git status --porcelain` empty.

**Compiled, on two toolchains.** Both objects were deleted first, gencode was
read from `compile_commands.json`, and real per-arch SASS was confirmed with
`cuobjdump --list-elf` rather than a PTX leg.

| TU | CUDA 12.6, sm_87 | CUDA 13.0, the full CI arch list |
|---|---|---|
| `cuda_mla_attn.cu` | `sm_87.cubin`, 1.56 MB | rc=0, **10 cubins**: 80, 86, 87, 89, 90a, 100a, 103a, 110, 120a, 121a |
| `cuda_flash_attn_fa2.cu` | `sm_87.cubin`, 514 KB | rc=0, **6 cubins**: 80, 86, 87, 89, 120a, 121a |

The second column reproduces what CI's `cuda-fat-build` asks, on the toolchain
that job uses. FA-2 being ON was MEASURED three ways rather than inferred from
the default: `CUDA feature fa2: ENABLED for [87]`, `VLLM_CPP_FLASH_ATTN:BOOL=ON`
in `CMakeCache.txt`, and the generated manifest
`VLLM_CPP_CUDA_FA2_COMPILED_ARCHS "87"`.

**Executed, and the assertion counts are the proof.** The same binaries produced
both columns. The control is `CUDA_VISIBLE_DEVICES=""`, so the delta is the
device and not the build.

| run | cases | assertions |
|---|---|---:|
| windowed decode alone, no device | 1 | **0** |
| windowed decode alone, on device | 1 | **49,158** |
| windowed prefill alone, no device | 1 | **0** |
| windowed prefill alone, on device | 1 | **467,010** |

`1 case / 0 assertions / SUCCESS!` is the shape both cases had worn until that
lease, and the right-hand column is the FIRST execution either has ever had.
Whole-binary figures on device: `test_ops_mla_attn` 246,290 to 2,401,528, and
`test_ops_mla_prefill` 329,772 to 2,931,678. Every filter matched exactly one
case, so no zero-match false green; every exit code was 0 and no run timed out.

**Numerically correct.** Windowed decode `MaxAbsDiff(gpu, cpu)` reads 2.38e-07,
2.68e-07 and 2.68e-07 across the three split arms, against a `< 1e-3` bar.
Windowed prefill, with 475 (query, key) pairs dropped across 57 queries, reads
`gpu_win` against `cpu_win` 0.00294137, `gpu_none` against `cpu_none` 0.00294137
and `gpu_win` against `expanded` 0.00294137, each against `< 3e-2`. The decisive
number is `gpu_win` against `gpu_none` = **1.06055**, against a `> 1e-2` bar: the
window demonstrably BITES on the device, so the FA-2 launcher is not dropping
`window_size` on the floor. That is the exact defect the `is_local`
normalization exists to prevent, now measured rather than argued.

**This run falsifies a label another row owns, filed as
[#2074](https://github.com/mudler/vllm.cpp/issues/2074).**
`cmake/CudaArchFeatures.cmake:347` labels the `fa2` row's Ampere `sm_8x` cells
"DERIVED+BUILD-VERIFIED ... NO Ampere board ran them here". Jetson AGX Orin IS
`sm_87`, one of those four cells, and the table above records FA-2 measured ON
and the prefill path executed on that board. The label is now wrong for `8.7`
and right for `8.0`, `8.6` and `8.9`, which one line cannot carry. Owner:
`BACKEND-CUDA-SM087` and
[`cuda-arch-ampere-fastpath.md`](cuda-arch-ampere-fastpath.md) WA-1, NOT this
row, so it is filed rather than fixed here.

**Two limits, so this is not read wider than it is.** Execution is proven on
**sm_87 ONLY**. The ten-arch result is **COMPILE-ONLY**, because CUDA 13 cannot
run against that box's NVRM 540.4.0 driver (`cudaGetDeviceCount rc=35`). And this
is KERNEL-level parity on two ops. It is not the end-to-end model gate, which is
unrelated and still owed.

**The two halves did NOT have the same amount of gate, and the first draft of
this paragraph read as though they did.** It named the two CUDA files together
and then said "the CUDA-vs-CPU window parity case is present in
`test_ops_mla_attn` and SKIPS without a device" — literally true of the DECODE
half and silent about the other. `test_ops_mla_prefill`'s two windowed cases
were CPU-only, and its only `HasCuda()` cases are pre-existing and unwindowed.
So the FA-2 MLA-prefill launcher's `is_local` block had NO case on ANY device,
and a later lease would have discharged the decode half against a record that
read as covering both. The #1969 review found this. The repair adds
`CUDA MLA prefill: the sliding window matches the CPU reference` to
`test_ops_mla_prefill`, mirroring the decode sibling, and comparing the windowed
device call against the windowed CPU op, against the unwindowed device call
(the window must BITE on the device, or the launcher could be dropping
`window_size` on the floor), and against the file's expanded single-query
oracle.

**On the CPU-only box where W4b-2 was written, NEITHER half executed, and the
assertion count is what said so.** Both cases skip without a device, and doctest
counts a case that returns before its first assertion as PASSED with ZERO
assertions. So `test_ops_mla_prefill` reads **7 cases / 329772 assertions** on
that box against 6 / 329772 before the repair: one more case and not one more
assertion. That measurement stands, and it is kept because it is the CONTROL for
the on-device column above, where the same case reads 467,010 assertions.
`## Owed` still names the prefill half separately from the decode half, because
W4b-2's first record merged them and the #1969 review caught it.

**Compilation was owed here too, and that was a separate statement.** Before the
lease, the only compile verification these two files could get on this change was
CI's `cuda-fat-build`, which had not reported on `fa96f9557` at the time of
writing. The `orin:gpu0` lease discharged compilation and execution together, so
neither statement is open.

**The fleet reading that scoped this debt, kept as dated evidence.** Measured
with `rc devices` on 2026-08-26, not taken from a report:

```
DEVICE     STATE                            HOLDER  ELAPSED  COMMAND
dgx:gpu0   unhealthy (no contact 1h27m50s)  -       -        -
orin:gpu0  unknown (no contact 1m17s)       -       -        -
thor:gpu0  unhealthy (no contact 1h16m32s)  -       -        -
```

**The conclusion drawn from that reading is SUPERSEDED, and it was wrong on two
counts.** It read that both CUDA hosts this row could use were QUARANTINED, and
it dismissed the third device with "it is an `orin` (sm_87) and not this row's
host in any case". Later the same day an `rc run` lease on `orin:gpu0` compiled
and executed both halves. The fleet recovered, and the device the paragraph
dismissed is the one that discharged the work. The observation above is
retained as evidence of the state at that timestamp. The conclusion built on it
is not. **This section asserts NO current fleet state.** A device state is a
reading at a moment, never a standing property, and a reading recorded here is
stale before the next reader arrives. Take a fresh `rc devices` reading before
you book a lease, and quote it raw beside its date. A 2026-08-27 reading stood
here and is REMOVED rather than corrected: it was paraphrased instead of quoted,
and this branch's fresh review read `thor:gpu0` as `unhealthy` on the same day.
An unquoted reading that a second observer contradicts is not evidence.

**And `thor:gpu0` could NEVER have discharged the PREFILL half, at any point.**
That makes §6.3's designation a design error on this path rather than a stale
reading. `cmake/CudaArchFeatures.cmake`'s `VT_CUDA_FEATURE_TABLE` carries the row
`fa2|8.0,8.6,8.7,8.9,12.0a,12.1a`, so `VT_FA2_ARCHS` resolves EMPTY for thor's
sm_110, `VLLM_CPP_FLASH_ATTN` is then never defined (`CMakeLists.txt`, at the
`if(VLLM_CPP_FLASH_ATTN AND VLLM_CPP_CUTLASS_HEADERS AND VT_FA2_ARCHS)` guard and
its `target_compile_definitions`), and `MlaPrefillAttentionCuda` throws instead
of computing (`src/vt/cuda/cuda_mla_prefill.cu:179-183`). The spec already
records the same fact in prose, in §7's **W0.5** phase entry and NOT in §6.3:
"Thor's MLA prefill throws rather than computes".

**§7's W0.5 entry already carried the executable evidence, and §6.3's
designation was never reconciled against it.** The W0.5 failure table lists FOUR
tests red on Thor under the cause "no vendored FA-2 — the build correctly
refusing what the arch lacks", and TWO of them are this brick's own gates:
`test_ops_mla_prefill` and `test_mla_attention_block`. Both the table and the
prose sentence live in §7's W0.5 entry, which is where a reader must go. So the
record showed the designated host failing the very binaries the CUDA half
needed, in the same document that designated it. This correction is therefore a
reconciliation of two statements the spec already held, not a new claim.

**The hosts that CAN serve this half are `orin:gpu0` (8.7) and `dgx:gpu0`
(12.1a)**, because both archs are in the `fa2` row and thor's 11.0 is not.
`orin:gpu0` can discharge BOTH of W4b-2's CUDA halves today: the gate is
weight-free and needs no checkpoint from the NAS. **Pick the CUDA host by
CAPABILITY, not by availability**: read the `fa2` row of the feature table
before booking a lease for anything on the FA-2 path.

### 4.9 W4b-3c lifts the `seq_len > index_topk` refusal, and the routing rule is upstream's own

**Scope.** Two units in one pull request, because the second makes the first
reachable at its own merge commit:

- **W4b-3c-1** — the two `vt` primitives the DSA sparse path needs, on CPU and
  CUDA: an OPTIONAL selected-slot arm on `vt::MlaDecodeAttention`, and a
  `vt` indexer op family (`vt::DsaIndexerLogits` + `vt::DsaTopkSelect`) that
  lifts W3's `std::vector<float>` host reference onto `vt::Tensor`.
- **W4b-3c-2** — the seam and the model: the indexer group on `MlaBlockDims` /
  `MlaBlockWeights`, the indexer call inside `mla::ForwardMlaAttentionBlock`,
  the per-token sparse-MQA routing on `MlaBlockMetadata`, the five indexer
  tensors in `MaterializeDots3NoteDevice`, and the NARROWED refusal.

**Out of scope, and it is a hard dependency rather than a preference.** The DSA
INDEX KV cache — the indexer's own 128-wide `k` for tokens computed on an
earlier step — is a SECOND attention group on the same layers, with its own
128-wide fp8 row and its own spec kind. Carrying more than one published group
is `KV-DSV4-MULTICACHE`
([#1925](https://github.com/mudler/vllm.cpp/issues/1925)), whose W3 generalized
path landed at `ca3dcda21` while this brick was in flight. **What that changed
is worth stating exactly, because the sentence this row carried before the
W4b-3c review described the PRE-`ca3dcda21` runner.** The runner picks the first
non-eagle `kFullAttention`/`kMlaAttention` group as its target
(`src/vllm/v1/worker/gpu/runner.cpp:703-712`); a leftover group beyond that
target, the GDN group and ONE `kFullAttention` draft slot switches it into the
multi-cache path (`:784-800`); and the `VT_CHECK` on `:860-870` then refuses
only four shapes — a spec that is neither an `AttentionSpec` nor a `MambaSpec`,
a SECOND recurrent group, an EAGLE draft group, and a group whose published
layer names do not all resolve to distinct in-range layer indices. **A second
non-eagle `AttentionSpec` group is now ALLOCATED, not refused.** So publishing
the indexer's key cache here would no longer throw; it would be allocated while
nothing on the model side read it, which is a worse failure than the refusal
this paragraph used to rely on. dots3-note publishes ONE uniform
`MLAAttentionSpec` for all 46 layers today, this brick does not touch
`MakeDots3NoteKVCache`, and #1925 still owns the capability. This brick therefore
serves the case whose index keys are all computed IN-STEP — a single-shot
prefill — and refuses the rest by name. Duplicating #1925 here is the failure
mode this paragraph exists to prevent.

#### The PIN-DELTA check, run before a line was ported

W3 transcribed the indexer maths at vLLM `06ecec7a84`, and
`deepseek_v4_dsa.h`'s own header block says `@ pin 555967922`. This row's
upstream is `bc2d63e650`. Those are three different revisions, so the
transcription was re-derived rather than trusted:

| Upstream file | `555967922` → `bc2d63e650` | Consequence here |
|---|---|---|
| `vllm/v1/attention/ops/triton_fp8_mqa_logits.py` | **byte-identical** (`git diff` empty) | `DsaIndexerLogits`'s source did not move; `:120-156` still holds |
| `vllm/model_executor/layers/sparse_attn_indexer.py` (the FOLD, `:203-207`) | **byte-identical, at the SAME line numbers** | `DsaIndexerWeightFold`'s source did not move |
| `vllm/model_executor/models/deepseek_v2.py::Indexer` | **one added line**, `assert cache_config is not None` (`:718`) | no numeric change; every other line of the class is byte-identical |
| `sparse_attn_indexer.py` (the top-k CALL SITE) | `:488` → `:509` | a LINE anchor moved; the SYMBOL `ops.top_k_per_row_prefill` did not |
| `sparse_attn_indexer.py` (elsewhere) | `+44/-10`, i.e. 54 lines TOUCHED and a NET +34: a `pcp` import move, `k: Tensor \| None`, a `dense_mha_metadata_layer_name` early-return, `compress_ratio` | scheduling and typing; none of it is the selection maths |

**Verdict: the maths is CURRENT and no re-port was owed.** What moved is one
line anchor, and this section re-cites at `bc2d63e650` throughout. That is the
outcome `porting.md` §"Name the symbol, not only the line" predicts, and it is
recorded as a measurement rather than as an assumption because the opposite
answer would have changed the design.

#### The upstream anchors, at `bc2d63e650`

| Ours | Upstream `file:line` @ `bc2d63e650` |
|---|---|
| `vt::DsaIndexerLogits` | `vllm/v1/attention/ops/triton_fp8_mqa_logits.py:120-156` (`dot`, `* kv_scale`, `ReLU`, `* weights`, sum over heads) |
| the weight FOLD | `vllm/model_executor/layers/sparse_attn_indexer.py:203-207` and `vllm/model_executor/models/deepseek_v2.py:840` (`weights * q_scale * softmax_scale * n_head_scale`) |
| `softmax_scale`, `n_head_scale` | `deepseek_v2.py:709` (`head_dim ** -0.5`), `:742` (`n_head ** -0.5`) |
| `vt::DsaTopkSelect` | `sparse_attn_indexer.py:509` (`ops.top_k_per_row_prefill`) + the short-context all-select |
| the indexer's `k_norm` | `deepseek_v2.py:708` — `LayerNorm(head_dim, eps=1e-6)`, i.e. mean-subtracting WITH a bias, not RMSNorm |
| the indexer's LEADING rope slice | `deepseek_v2.py:804-806`, `:813-817` (`q_pe, q_nope = split(q, [rope_dim, head_dim - rope_dim])`) |
| the indexer's rope POLARITY | `deepseek_v2.py:1155-1160` — `is_neox_style = not indexer_rope_interleave`, over the SAME `qk_rope_head_dim`, `max_position_embeddings` and `config.rope_parameters` as the main MLA rope at `:1104-1109` |
| the indexer CALL SITE | `vllm/models/dots3_note/nvidia/model.py:171-172` — between the MLA rope and `mla_attn` |
| the sparse per-token MQA | `vllm/models/dots3_note/nvidia/attention.py:744-815` `Dots3NotePaddedSparseImpl.forward_mqa` |
| **the ROUTING RULE** | `vllm/model_executor/layers/attention/mla_attention.py:825-851` + `vllm/model_executor/layers/attention/sparse_mla_attention.py:296-299` |

**Uniqueness was re-measured at `bc2d63e650`, and the blanket claim this
section first made was FALSE.** The claim was "every anchor above is `grep -c`
== 1 on its symbol"; the W4b-3c review found six that are not, and a false
uniqueness claim is worse than none — a reader trusts it and stops checking. So
the claim is now the measurement, with the six named and each given the
discriminator that picks the cited occurrence:

| symbol | occurrences at `bc2d63e650` | what picks the cited one |
|---|---|---|
| `input_precision="ieee"` | `triton_fp8_mqa_logits.py:125`, `:146` | both are INSIDE the cited range `:120-156`; the range is the anchor, and it covers the kernel's two branches |
| `tl.maximum(scores, 0.0)` | `triton_fp8_mqa_logits.py:129`, `:150` | the same range, the same two branches |
| `is_neox_style=False` | `deepseek_v2.py:548`, `:1108` | `:548` is `DeepseekV2Attention` (class at `:450`); the cited `:1108` is `DeepseekV2MLAAttention` (class at `:982`) |
| `assert cache_config is not None` | `deepseek_v2.py:718`, `:1152` | the cited `:718` is inside `class Indexer` (`:667`); `:1152` is `DeepseekV2MLAAttention.__init__` |
| `triton_convert_req_index_to_global_index` | `attention.py:33`, `:760` | `:33` is the IMPORT; the cited `:760-767` is the CALL |
| `def forward_mqa` | `attention.py:656`, `:744` | `:656` is `Dots3NoteTritonMLAImpl` (`:439`); the cited `:744` is `Dots3NotePaddedSparseImpl` (`:697`) |

Every other anchor in this section IS `grep -c` == 1 on its symbol. The reason
the check exists at all is that this row has had a line anchor go stale inside a
single pull request twice; the reason it is now written as a measurement is that
a summary of a check nobody can reproduce is not a check.

#### The routing rule is upstream's, and it is why the EXISTING path does not move

The temptation was to route every full-attention token through sparse MQA.
Upstream does not, and the condition is one line
(`sparse_mla_attention.py:296-299`):

```python
use_dense_mha=(prefill_max_seq_len <= self.topk_tokens
               and not ...attention_config.sparse_mla_force_mqa)
```

and `mla_attention.py:829-851` consumes it: `if self.impl.is_sparse and
num_mha_tokens > 0` and `not use_mha`, then `num_mqa_tokens = q.size(0)` — ALL
tokens go MQA. So:

- `prefill_max_seq_len <= index_topk` — the top-k selects every causal
  candidate — keeps the DENSE MHA prefill. **That is exactly what W4a/W4b-2
  already do, so the path this row already gates does not move a byte.**
- `prefill_max_seq_len > index_topk` — the selection actually prunes — switches
  the WHOLE step to per-token MQA, one query per token, over its own selected
  key list.

Mirroring that rule rather than inventing one is what keeps §4.6's and §4.8's
gates valid, and it is why the `## Owed` entry for the selection can close
without a second numerics story.

#### What the NARROWED refusal is

From "any `seq_len > index_topk`" to "a STEP in which some request needs a
selection and some request has CACHED CONTEXT (`num_computed_tokens_cpu[i] >
0`)". The discriminator is the index KV cache, not the sequence length: the
indexer's `k` for a token is produced by `wk_weights_proj` from that token's
hidden state, so a step that computes every token of every sequence has every
index key in hand, and a step in which anything resumes does not.
`CommonAttentionMetadata::num_computed_tokens_cpu` already carries the
discriminator. A single-shot prefill of a long prompt is therefore served
correctly and SPARSELY; a step that resumes and also prunes refuses, and names
#1925.

**THE UNIT IS THE STEP, and the first draft of this brick got that wrong.** It
wrote the refusal PER REQUEST — `seq_len > index_topk AND computed > 0` on each
one — while `BuildDots3NoteSparseStep` disabled the sparse route for the WHOLE
step the moment ANY request resumed, because the indexer's key space is the
step's own tokens (`indexer_cu_seqlens_q`). Two different predicates, and the
gap between them was reachable by continuous batching's most ordinary shape:
`{one resumed request at or under index_topk, one FRESH prompt past it}` passed
the refusal and took no sparse route, so it was served DENSE. See "The
mixed-batch defect" below. Both questions are now answered by one function,
`Dots3NoteSparseEligibilityOf`, and the invariant is stated in one line: a step
that prunes is either served sparsely or REFUSED, with no third outcome.

#### The `vt` primitives, precisely

**(1) `MlaDecodeAttentionArgs` gains an optional selected-slot pair.** Mirrors
the shape `window_size` already has:

```cpp
const Tensor* topk_indices = nullptr;   // [batch, topk] i32, TOKEN POSITIONS, -1 = none
const Tensor* valid_counts = nullptr;   // [batch] i32
```

Absent (both null) is a NOT-TAKEN branch, not a mask: the kernel's key loop
either walks `[j_start, seq_len)` or walks the selected list, and nothing else
changes. The kernel keeps the existing `blk = j / block_size` block-table walk,
which is our equivalent of upstream's
`triton_convert_req_index_to_global_index` (`attention.py:760-767`) done INSIDE
the kernel rather than ahead of it — so no flat-cache `as_strided` view
(`attention.py:792-795`) is needed, and no second copy of the index buffer.

`ops.cpp` refuses BY NAME: one tensor present without the other; a non-i32
dtype; a wrong rank or shape; a `topk` of 0; a count exceeding `topk`; and a
window AND a selection together, which upstream cannot produce because the
sliding layers set `self.indexer = None` (`model.py:432-434`).

**The count check is host-side only, and that is a recorded deviation.**
Reading `valid_counts` in the validator would force a device synchronization on
every decode step, which is a per-step cost on the model path. So `ops.cpp`
refuses an over-large count when the tensor is host-readable, and BOTH kernels
clamp `min(count, topk)` so an over-large count can never read out of bounds.

**(2) `vt::DsaIndexerLogits` / `vt::DsaTopkSelect`.** The `vt::Tensor` form of
`DsaIndexerLogits` / `DsaTopkSelect`, which stay as the host oracle. The fold is
`args.softmax_scale * args.n_head_scale` times an OPTIONAL per-(token,head)
`q_scale` (absent ⇒ 1, the unquantized arm). `k_norm` and the leading rope slice
are NOT new ops: they route through `vt::LayerNorm` (weight + bias) and
`vt::RopeFromCache` (`rotary_dim < head_dim`, `is_neox_style` from the config)
over a strided view. Writing a second copy of either would be the parallel path
AGENTS.md forbids.

#### The gate design, and why a single tolerance says almost nothing here

**The op-level oracle is the op itself on a different input** — the trick W4b-2
proved (§4.8). A sparse call over a paged sequence is compared against an
UNWINDOWED, UNSELECTED call on a freshly built single-page cache holding exactly
the selected keys.

**The identity case is the strongest assertion available**, and it is asserted
BIT-FOR-BIT: a selection listing EVERY causal key must equal no selection at
all, byte for byte, on both backends. A mask applied after the fact cannot pass
that, and neither can a selected path whose split partition differs from the
dense one.

**The selection is DISCRETE, so its error is bimodal** — either a slot flips and
the residue jumps to mechanism scale, or it does not and the residue is the
float floor. A single tolerance is therefore nearly uninformative. §4.5 measured
this row's strict selection margins at **1.29e-3** with float logits (~1e-7
error, three orders of headroom), while a **bf16** logit of order 1 carries
**~4e-3** — LARGER than that margin. A fixture inherited from a continuous gate
would be a coin flip. So the gate additionally:

1. asserts **selection-set equality** against the reference as its own discrete
   assertion;
2. **prints the minimum decision margin** and requires it to exceed a stated
   multiple of the working precision's ulp, so the fixture's adequacy is
   MEASURED rather than assumed;
3. **prints how many query rows actually prune and how many keys are dropped,
   by number** — below the selection threshold every assertion passes on an
   implementation that performs no selection at all;
4. keeps deliberate EXACT ties (equal in both precisions, broken by the same
   smaller-index rule) and ensures no NEAR-tie sits on the k-th boundary.

#### Predicted-GREEN mutations, named in advance

`weights * q_scale * softmax_scale * n_head_scale`: `softmax_scale` and
`n_head_scale` are **global positive scalars**, so dropping either cannot move
an argmax. Their mutations read GREEN **definitionally**, not because the gate
has a hole. `q_scale` is per-(token,head) and is 1 on an unquantized arm, so it
is inert here and live only in fp8. All three are predicted in the mutation
table and labelled as such, and the folded logits are additionally asserted
**by value** so the fold is covered anyway.

#### Reachability

This unit has a production call site and does NOT take the staged-slice
exception. The chain is `ModelRegistry::Resolve` → `load_weights` over a real
`SafetensorsFile` → `ModelRegistry::Forward` →
`Dots3NoteModel::ForwardDevice` → `mla::ForwardMlaAttentionBlock` → the indexer
ops and the selected-slot MLA decode. The smallest failing test enters through
that chain, not by constructing the type. The reachability mutation deletes the
production call site in a scratch copy and requires the focused gate to go RED.

#### Risks

| Risk | Control |
|---|---|
| the seam has four callers, one SACRED (DeepSeek-V2) | the indexer group is EMPTY by default and every branch is not-taken; a six-arm byte-identity probe is rebuilt in this session at this base SHA, and the neutrality is mutation-proven by LEAKING the indexer onto the shared path and requiring RED |
| a fixture whose margins are below bf16 ulp | the printed minimum margin, gated against a stated ulp multiple |
| a gate that passes on an implementation that never selects | the printed prune counts and dropped-key counts |
| a CUDA arm written and not run | executed under an `rc run` lease with a `CUDA_VISIBLE_DEVICES=""` control on the SAME binaries; the assertion-count delta is the proof |
| duplicating #1925 | the narrowed refusal, which is why the refusal still exists |

#### The gate, met

| Gate | Result |
|---|---|
| `test_ops_dsa_indexer` (NEW) | 10 cases / 176 assertions, CPU |
| `test_ops_mla_attn` | 23 cases / 289,324 assertions (from 15 / 246,290 at base) |
| `test_dots3_note_attn` | **43 cases / 3,942 assertions** after the `!well_formed` arm got its own case (42 / 3,803 after the review repair; 40 / 3,558 before it; 36 / 3,037 at base) |
| `test_mla_attention_block` | 13 cases / 2,247,730 assertions |
| `test_deepseek_v2_forward` (SACRED sibling) | 11 cases / 1,052 assertions |
| `test_dots3_note_scaffold` | 26 cases / 110,819 assertions |
| `test_deepseek_v4_dsa` (the host oracle) | 13 cases / unchanged |
| `test_ops_mla_prefill` / `_chunked_context` / `_absorb` | 7 / 5 / 9 cases, unchanged |
| `test_kimi_linear_forward`, `test_minicpm3_paged_engine` | 15 / 1 cases, unchanged |

**The end-to-end numbers, printed by the gate rather than quoted here from
somewhere else.** At `index_topk` 2 against a 6-token prompt: **4 of 6** query
rows really prune and **10** causal keys are dropped; the device sits **0.00916
relative** from the SELECTING reference against a 0.05 bound (5.46x headroom)
and **0.722 relative** from the NON-selecting one — 79x the residue, so the
selection demonstrably bites. The indexer-rope-polarity case measures 0.0183
against the NeoX reference and 0.707 against the GPT-J one, a 38.7x separation.

**The op-level numbers.** The MLA selection case reports 4 of 4 rows pruning and
115 keys dropped over a 38-wide topk row. The indexer case reports 5 of 8 rows
pruning, 15 keys dropped, **2 boundary decisions decided by an EXACT tie** and 3
by a strict margin, with a **minimum strict margin of 0.988 against a bf16 ulp
of 0.0202** at that fixture's own logit scale — 48.8x, against a stated bar of
4x.

**A fixture defect the discipline caught, and the repair was the fixture.** The
first version of `test_ops_dsa_indexer.cpp` sampled `k` from uniform noise and
measured a minimum strict margin of **7.43e-4 against a bf16 ulp of 1.28e-3** —
a fixture whose selection the model path's own operand narrowing could have
flipped, and which nonetheless passed because it happened not to. That is §4.5's
1.29e-3-against-4e-3 warning reproduced at a smaller scale. The threshold was
NOT relaxed; the fixture was replaced by a designed one whose margins are
analysable.

#### The six-arm DeepSeek byte-identity probe, COMMITTED and measured

`mla::ForwardMlaAttentionBlock` has four callers, one of which is DeepSeek-V2
with a SACRED gate. The probe is now committed beside the DeepSeek gates
(`tests/vllm/model_executor/layers/attention/test_mla_attention_block.cpp`), so
the next brick can reproduce it rather than write a third one. Measured in ONE
session with the probe injected byte-for-byte into a `git archive` of the base:

| Arm | base `157636cf1` | head |
|---|---|---|
| A1 V2-Lite bf16 decode-only | `a2f1e41a168210a8` | `a2f1e41a168210a8` |
| A2 V2-Lite bf16 prefill-only, no context | `278156e492ef2281` | `278156e492ef2281` |
| A3 V2-Lite bf16 prefill WITH chunked context | `232c61867237916e` | `232c61867237916e` |
| A4 V2-Lite bf16 MIXED, decode packed first | `1e0874090a29a4fa` | `1e0874090a29a4fa` |
| A5 V2-Lite f32 MIXED | `85d76ad77adbbb47` | `85d76ad77adbbb47` |
| A6 V3 q_lora bf16 MIXED | `82d987ccac222326` | `82d987ccac222326` |

Each arm also asserts run-to-run stability, so a printed fingerprint is a
property of the TREE rather than of one run.

#### The CUDA arm EXECUTED, with a same-binary control

`dgx:gpu0`, NVIDIA GB10, driver `580.173.02`, compute capability 12.1, inside an
`rc run` lease — job `f3600dc2-2d80-4b01-8dad-6f9571616d68`, `--max-runtime
120m`. No `ssh`. CUDA toolkit 13.0 V13.0.88, apt-installed per run from
`developer.download.nvidia.com/.../ubuntu2404/sbsa`. The build was gated on a
smoke test that COMPILES AND LAUNCHES a kernel and checks the value
(`SMOKE=OK value=4242`), because a toolkit that compiles for an architecture its
driver cannot run is a shape this fleet has produced before.

| Binary | on the device | `CUDA_VISIBLE_DEVICES=""`, SAME binary | delta |
|---|---|---|---|
| `test_ops_mla_attn` | 22 cases / **2,516,250** assertions | 22 cases / **287,274** | +2,228,976 |
| `test_ops_dsa_indexer` | 10 cases / **242** | 10 cases / **176** | +66 |

Per CUDA case, which is what says each one RAN rather than returning before its
first assertion — a doctest case that early-exits is scored PASSED with ZERO
assertions, a skip wearing a pass:

| Case | device | control |
|---|---|---|
| `CUDA mla_decode: the DSA selection matches the CPU reference` | **49,159** | 0 |
| `CUDA mla_decode: a FULL selection is BIT-IDENTICAL to no selection` | **24,579** | 0 |
| `CUDA dsa_indexer: the SELECTION SET is identical to the CPU arm` | **63** | 0 |
| `CUDA dsa_indexer: BF16 operands select the IDENTICAL set on the device too` | **3** | 0 |

`cuobjdump --list-elf` reads `cuda_dsa_indexer.cu.1.sm_121.cubin` and
`cuda_mla_attn.cu.1.sm_121.cubin`. The archive's sha256 was verified inside the
lease, and every file the CUDA arm covers is byte-unchanged between that archive
and this head.

**The scope of that result, stated rather than left to be assumed.** Execution
is proven on **sm_121 only**. This is kernel-level parity on four op cases, not
the end-to-end model gate, which remains the first entry under `## Owed`.

#### A NaN the reading found, and the device proof that the guard is load-bearing

`MlaDecodeStage1` scores a dead selected slot `-inf` and then rescales by
`expf(m - m_new)`. When EVERY slot of a `kNTile` run is dead, `m_new` stays at
the empty row's `-inf`, `-inf - -inf` is NaN, and `expf(NaN)` poisons the
accumulator for the whole (request, head). Unreachable before this wave — every
key of a contiguous range has a finite score — and reachable with it, because
the op's contract makes `-1` the "no token" sentinel and a caller may pad INSIDE
its own `valid_counts`. The CPU arm never had the tile: it `continue`s a dead
slot. So the two arms disagreed on an input the contract admits, a number
against a NaN.

**Found by reading the kernel, not by a failing gate**, and stated that way
because the distinction is what a reviewer needs. The case that reaches it is
shaped for the boundary: a count of 12 whose first 8 entries are `-1`, with
`num_kv_splits = 1` so the tiles are exactly `[0,8)` and `[8,12)`.

Proven on the device in a second lease, job
`fa0e3f42-87f9-4f4a-b510-8f4386aace16` on `dgx:gpu0`:

| tree | build | `test_ops_mla_attn` on the device | the case alone |
|---|---|---|---|
| as committed | rc 0 | 23 cases / **2,522,398** assertions, SUCCESS | 1 case / 6,148, SUCCESS |
| guard REMOVED | rc 0, 0 compile errors | — | 1 case / 2,053, **FAILURE** |

The mutation BUILT, so the red is a result rather than a stale binary reading as
a pass.

#### The mutation table

Every row was driven through the committed `scripts/mutation-harness.py`, which
refuses an absent anchor, refuses a dirty tree, restores byte-for-byte with a
sha256 check, and prints the compiler exit beside every row. `PG` marks a row
whose greenness was PREDICTED IN ADVANCE.

| id | mutation | gate | cc exit | verdict | cases/asserts failing |
|---|---|---|---:|---|---|
| M1 | drop `softmax_scale` from the fold **(PG for the SELECTION)** | `test_ops_dsa_indexer` | 0 | DETECTED, by VALUE only | 2 cases |
| M2 | drop `n_head_scale` from the fold **(PG for the SELECTION)** | `test_ops_dsa_indexer` | 0 | DETECTED, by VALUE only | 2 cases |
| M3 | drop the ReLU from the MQA logit | `test_ops_dsa_indexer` | 0 | DETECTED | 3 cases / 12 |
| M4 | drop the per-head gate weight from the logit | `test_ops_dsa_indexer` | 0 | DETECTED | 2 cases / 33 |
| M5 | break the tie rule: LARGER key index wins | `test_ops_dsa_indexer` | 0 | DETECTED | 2 cases / 6 |
| M6 | emit the selection in RANK order, not ascending key order | `test_ops_dsa_indexer` | 0 | DETECTED | 1 case / 5 |
| M7 | drop the out-of-window `-inf` fill (the causal mask) | `test_ops_dsa_indexer` | 0 | DETECTED | 1 case / 56 |
| M8 | short-context branch emits a PARTIAL list | `test_ops_dsa_indexer` | 0 | DETECTED | 2 cases / 13 |
| M9 | IGNORE the selection: walk the contiguous range | `test_ops_mla_attn` | 0 | DETECTED | 2 cases |
| M10 | keep the list, IGNORE `valid_counts` | `test_ops_mla_attn` | 0 | **SURVIVED, then DETECTED** | 1 case / 2048 |
| M11 | drop the `-1` sentinel skip | `test_ops_mla_attn` | 0 | DETECTED | 1 case / 2048 |
| M12 | drop the "both or neither" selection-pair refusal | `test_ops_mla_attn` | 0 | DETECTED | 1 case |
| M13 | drop the window-plus-selection refusal | `test_ops_mla_attn` | 0 | DETECTED | 1 case / 1 |
| M14 | drop the over-large `valid_counts` refusal | `test_ops_mla_attn` | 0 | DETECTED | 1 case / 1 |
| M15 | **REACHABILITY**: delete the sparse route's production call site | `test_dots3_note_attn` | 0 | DETECTED | 1 case / 2 |
| M16 | drop `dims.has_indexer()` from the run predicate | `test_dots3_note_attn` | 0 | SURVIVED, see below | — |
| M17 | `k_norm` as an RmsNorm: no mean subtraction, no bias | `test_dots3_note_attn` | 0 | DETECTED | 1 case / 2 |
| M18 | the `k_norm` epsilon becomes the model's `rms_norm_eps` | `test_dots3_note_attn` | 0 | SURVIVED, see below | — |
| M19 | the indexer rope pairing follows the MAIN rope | `test_dots3_note_attn` | 0 | **SURVIVED, then DETECTED** | 1 case / 2 |
| M20 | the indexer ropes the TRAILING slice, not the leading one | `test_dots3_note_attn` | 0 | DETECTED | 1 case / 2 |
| M21 | the resumed-request refusal is deleted | `test_dots3_note_attn` | 0 | DETECTED | 2 cases / 3 |
| M22 | the sparse route fires even when NOTHING prunes | `test_dots3_note_attn` | 0 | SURVIVED, see below | — |
| M23 | the sparse route fires on a RESUMED step too | `test_dots3_note_attn` | 0 | SURVIVED, see below | — |
| M24 | per-token `seq_lens` become the request's FULL length | `test_dots3_note_attn` | 0 | SURVIVED, see below | — |
| M25 | the indexer group is not loaded on FULL layers | `test_dots3_note_attn` | 0 | DETECTED | 4 cases / 4 |
| M26 | **LEAK**: the indexer runs unconditionally, both predicates ignored | `test_mla_attention_block`, `test_deepseek_v2_forward` | 0 | DETECTED | 6 of 13 cases, and 5 of 11 on the SACRED sibling |
| M27 | remove the all-dead-tile NaN guard **(ON THE DEVICE, `dgx:gpu0`)** | `test_ops_mla_attn` | 0 | DETECTED | 1 case / 2 |

**Three rows died on `-Werror=unused-variable` on their first form and were
re-run with the `(void)` shape** — M7, M9 and M10. A non-building mutation is
`NOT A RESULT`, and the harness marks it `BUILD_FAILED` rather than scoring it,
which is the only reason those three are not silent survivors here.

**M1 and M2 confirm the prediction rather than contradict it.** They were
predicted green FOR THE SELECTION, because `softmax_scale` and `n_head_scale`
are global positive scalars and cannot move an argmax. Measured: under M1 the
two failing cases are `reproduces the W3 host reference exactly` and `one logit
BY VALUE, from first principles`, and every SELECTION case — the set equality,
the bf16 arm and the tie rule — stays GREEN. The fold is covered by value, which
is exactly why that assertion was written.

**M10 SURVIVED, and the repair was the gate.** Every selection case padded a
row's tail with `-1`, which the kernel skips anyway, so `valid_counts` was never
tested. Upstream's topk buffer is a persistent workspace reused across steps
(`sparse_attn_indexer.py:431-432` — `:426-430` is the comment ABOVE the
statement, which is what this row first cited — narrowed at `attention.py:759`),
so the slots
past a row's live count can hold a PREVIOUS step's real positions. A case that
pads the tail with real, in-range, causally valid keys closed it, with a control
showing that honouring the tail IS a different answer. DETECTED after.

**M19 SURVIVED, and the repair was the gate.** On the released config
`indexer_rope_interleave` resolves so that BOTH ropes are GPT-J, so reading one
flag from the other is invisible — a property of the fixture, not of the
mechanism. A case that drives the two apart closed it. DETECTED after.

**Four rows are green BY CONSTRUCTION, and each says why.**

- **M16** drops `dims.has_indexer()` and keeps the metadata half. It survives
  because only FULL layers ever receive sparse metadata, so the two conditions
  are redundant ON THE CURRENT ROUTING and the dims half is defence in depth.
  The predicate AS A WHOLE is load-bearing, and M26 measures that: forcing it
  true reds 6 of `test_mla_attention_block`'s 13 cases **and 5 of the SACRED
  `test_deepseek_v2_forward`'s 11**, which this section under-reported until the
  W4b-3c review measured the second gate too.
- **M18** moves the `k_norm` epsilon from the upstream literal `1e-6` to the
  bench's `rms_norm_eps` of `1e-3`. MEASURED: the sparse case's residue is
  unchanged to six significant figures (0.00916328 either way). The epsilon sits
  inside the sqrt of a LayerNorm over a variance of order 1, so three orders of
  magnitude move `rstd` by ~5e-4 relative — inside the bf16 store's own 2^-8
  granularity, and far inside the 0.0417 minimum selection margin. This is the
  §4.5 hazard in its device form: transcribing the literal from
  `deepseek_v2.py:708` is what makes it right, and no value gate on a bf16 path
  can hold it. Owed.
- **M22** fires the sparse route when nothing prunes. It survives because a
  selection naming every causal key computes the SAME function as dense
  attention — that identity is what this whole brick rests on and what the
  op-level identity case asserts bit-for-bit. What the mutation changes is the
  ROUTE (per-token MQA instead of the MHA prefill), which is a divergence from
  upstream's `use_dense_mha` visible as behaviour and cost rather than as wrong
  tokens.
- **M23** fires the sparse route on a resumed step. **The justification this
  row first wrote here was TRUE ONLY FOR `num_reqs == 1`, and it is the sentence
  that made the W4b-3c defect look covered.** It read: "a resumed request past
  `index_topk` is refused before the route is consulted, and one under
  `index_topk` does not prune, so the route is not taken either way". Both
  clauses are per-request statements about a decision that is made per STEP. In
  a batch of `{resumed request under index_topk, fresh prompt past it}` the
  first request kept the route off for the WHOLE step while the second one
  needed it, and neither clause fired — the step was served DENSE with no
  message. The repair widened the refusal to the exact complement of
  `Dots3NoteSparseEligibility::Active`, so a step that prunes is either served
  sparsely or refused; M21 and the new mixed-batch case are what hold it, and
  M23 stays green because with the refusal in place the mutated route is
  unreachable for every batch shape, not only for single-request ones.
- **M24** widens the per-token `seq_lens` to the request's full length. On the
  sparse route `seq_lens` is an out-of-range GUARD rather than a selector — the
  key list is already causally bounded by `win_end[t] = t + 1` — so widening it
  weakens a guard without changing an answer. The guard's own violation is what
  M11 measures.

#### The mixed-batch defect the fresh review found, and the repair

**What it was.** The refusal fired on `seq_len > index_topk AND computed > 0`
per request (`dots3_note_device.cpp`), while `BuildDots3NoteSparseStep` returned
inactive for the WHOLE step the moment any request had `computed > 0`. A batch
of `{one resumed request with seq_len <= index_topk, one fresh prefill with
seq_len > index_topk}` satisfied neither predicate: no refusal, no sparse route,
dense attention on a sparse model, silently. Every dots3-note device case in the
brick was `num_reqs = 1`, so no gate and no mutation could see it.

**Measured before the repair, at the repair branch's own head**, through the
real chain (`ModelRegistry::Resolve` -> `load_weights` over a real
`SafetensorsFile` -> `ModelRegistry::Forward`), `num_reqs = 2`, `index_topk` 2,
request 0 resumed (`seq_lens` 2, `computed` 1, packed first as the seam
requires), request 1 a fresh 5-token prefill. Request 1's own logit rows compared
against BOTH references:

| comparison | relative | verdict |
|---|---|---|
| device vs the SELECTING reference | **0.880079** | 17.6x OUTSIDE the 0.05 bound |
| device vs the NON-selecting reference | **0.0103532** | INSIDE the bound |

That is this brick's own gate inverted: the device agreed with the reference
that performs no selection. The fresh review measured the same polarity on its
own fixture (0.57867 selecting / 0.0178515 dense).

**The fix, and why this one.** Upstream's routing decision is per STEP —
`use_dense_mha = prefill_max_seq_len <= self.topk_tokens`
(`sparse_mla_attention.py:296-299` @ `bc2d63e650`, re-derived at this head) and
`mla_attention.py:849-851` promotes `num_mqa_tokens = q.size(0)`, i.e. the whole
step INCLUDING its decode tokens. Upstream can promote a resumed request because
it caches the indexer's keys; we cannot, and that cache is #1925. So the two
predicates were merged into one function, `Dots3NoteSparseEligibilityOf`, and
the refusal became its exact complement. **This is the conservative option of
the two the review named, chosen deliberately**: per-request routing is
expressible, but it rescues only the sub-case in which every resumed request is
at or under `index_topk`, and a resumed request PAST it still needs #1925 either
way. **BOTH sub-cases are ordinary** — at the released `index_topk` of 2048 a
co-scheduled decode under 2048 tokens is at least as common as one over it — so
what the refusal turns away today is a common serving shape and not a corner.
Refusing is still correct; serving the wrong tokens is not. The per-request
route and the cost of refusing are recorded under `## Owed`.

**RED before, GREEN after**, both measured with the build exit and the run exit
captured separately:

| | `test_dots3_note_attn` |
|---|---|
| new cases against the PRE-repair code | 2 cases, **1 failed**, 2 assertions failed: `CHECK_THROWS_WITH_AS(...) did NOT throw at all!` on both the `index_topk` and the `#1925` arm |
| the same cases after the repair | 42 cases / 3,803 assertions, 0 failed |
| after the `!well_formed` arm got its own case (below) | **43 cases / 3,942 assertions**, 0 failed |

**The refusal has TWO arms, and only one of them had a case.** The repaired
refusal is `prunes && (resumes || !well_formed)`, and the fresh review that
passed this brick reached the `!well_formed` arm only with a throwaway probe:
nothing committed entered it, so the message's ternary could lose its false arm
entirely and every gate in the tree stayed green. That is the same shape as the
defect this brick spent two review rounds repairing — a predicate nothing
measures — so a third case now holds it: `a step whose metadata is NOT SHAPED
the way the sparse route reads it REFUSES by name`. ONE FRESH 5-token request
past `index_topk` with a `query_start_loc` one entry too long, so `prunes` is
true and `resumes` is false; the same request on WELL-FORMED metadata is the
control and runs. RED-before was measured by MUTATION at this head, each anchor
`grep -c` == 1 before it was applied, the compiler exit captured separately from
the run exit, and the file restored and `md5sum -c` verified after each:

| mutation | build | run | result |
|---|---|---|---|
| the false arm's string emptied (`std::string("")`) — "delete the false arm" | exit 0 | exit 1 | **1 case, 1 assertion failed**, `Contains("not shaped")` against a message reading `... PRUNES (model.py:171) — and .` |
| the `VT_CHECK` condition widened to `!prunes \|\| !resumes \|\| Active()`, so only the `resumes` arm refuses | exit 0 | exit 1 | **1 case, 2 assertions failed**, both arms `threw a DIFFERENT exception` |

The second mutation also measured something worth recording: with the dots3-note
refusal gone, this shape does NOT reach a wrong answer — it throws later, at
`deepseek_v2.cpp:778`'s generic `query_start_loc must have num_reqs + 1
entries`. So what the `!well_formed` arm buys is the MESSAGE and not the safety
on this particular violation, and the case gates it as such.

The first two cases are `TWO fresh prompts past index_topk in ONE step are each
served SPARSELY` — `num_reqs = 2`, device **0.00916328** relative from the selecting
reference against the 0.05 bound and **0.872793** from the non-selecting one,
7 of 11 rows pruning and 16 causal keys dropped — and `a MIXED step ... REFUSES
rather than silently serving dense`, which carries two controls: the same
resumed request beside a SHORT fresh prompt still runs, and two FRESH requests
one of which is past `index_topk` still run. Without those controls the case
would pass on "any two-request step refuses".

#### The UBSan regression three plain gates could not see

**What CI found.** `sanitize-cpu (address,undefined)` went RED on this branch's
head while every plain gate stayed green:

```
src/vt/cpu/cpu_layernorm.cpp:33:60: runtime error: load of misaligned address
0x7fed2fc1de81 for type 'short unsigned int', which requires 2 byte alignment
```

The control is that the same lane was `success` on W4b-2's merge commit
`21fe11cf1`, so the finding is this brick's.

**What it actually was, measured rather than assumed.** The misaligned tensor is
`w.indexer_k_norm_weight`, and the load is `mla_attention.cpp:621`'s
`vt::LayerNorm` — the ONE call this brick added that hands a CHECKPOINT weight to
`vt::LayerNorm`. `LoadBf16Direct` does not always copy: since ENG-LOAD-DIRECT-UPLOAD
([#150](https://github.com/mudler/vllm.cpp/issues/150)) a whole-range verbatim
weight BORROWS the mmap'd safetensors payload (`BorrowStTensorBytes`,
`qwen3_5_weights.cpp:431`), so `Tensor::data` points into the file. A safetensors
tensor's address is the running byte total of every tensor ahead of it added to
the `8 + header_len` prologue, and neither term is padded here, so it can be
**odd**. `src/vt/cpu/cpu_layernorm.cpp`'s local `LoadF32At` read it through a
`uint16_t*`, which is undefined behaviour even on x86, where the load executes
and returns the right bytes.

**THE BRIEF'S FIRST DIAGNOSIS WAS WRONG, and the record says so because the
wrong one is the plausible one.** The natural reading of "misaligned bf16 view"
is a `Slice` at an odd byte offset, or arithmetic on this row's padded physical
latent row. It is neither. There is no producer defect: an arbitrary-address
tensor payload is this tree's SETTLED contract. `include/vt/unaligned.h` states
it ("mmap-backed tensor payloads may begin at any byte offset, so forming a
typed pointer to them is undefined"), `ea4deb203` ("Use defined arbitrary-address
tensor reads") gave `vt::cpu::LoadF32` (`cpu_ops.cpp:32`) exactly this shape for
exactly this reason, and
[#627](https://github.com/mudler/vllm.cpp/issues/627) is the open issue that
owns the class — its title is already "three recurrences found by UBSan". This
is recurrence four, at a site written after `ea4deb203` that did not inherit it.

**The fix is therefore at the READER, and it is `ea4deb203` replayed**:
`cpu_layernorm.cpp`'s `LoadF32At` now goes through `vt::LoadUnaligned`, which is
a `memcpy` — the same bytes, the same value, **no numerics moved**, which the
unchanged assertion counts below confirm. The STORE side keeps its typed
pointer, mirroring `vt::cpu::StoreF32` (`cpu_ops.cpp:43`): an output is an
engine-allocated buffer and never a borrowed mapping. One `LoadF32At` serves all
five ops this translation unit registers (LayerNorm, Relu, GeluTanh, GeluErf,
Add), so the repair covers them together.

**An alignment REFUSAL in `vt::LayerNorm` was considered and rejected.** A
`VT_CHECK` that a bf16 input is 2-byte aligned would turn this UBSan-only finding
into an always-on message, but it would refuse a LEGITIMATE input: the borrowed
weight is what `LoadBf16Direct` is supposed to produce, and the refusal would
make the indexer's `k_norm` unloadable on any checkpoint whose data section
starts odd. It would also contradict `ea4deb203` and #627 in the same breath.
The seam's contract is that a reader accepts an arbitrary address; the defect
was a reader that did not.

**WHY THREE INDEPENDENT GATES MISSED IT, which is the transferable part.** The
implementer, the fresh reviewer and the operator each ran the full gate and each
was green, because **no gate in this repository's ordinary loop enables a
sanitizer**. `scripts/agent-preflight.sh` and the plain `ctest` build with the
default flags, where an unaligned 2-byte load is a working instruction and the
tokens are byte-identical. The detector exists — it is a CI lane, `sanitize-cpu`,
configured with `-DVLLM_CPP_SANITIZE='address,undefined'` — and it is the only
thing in the loop that can see this class. A local reproduction is one configure
away and is now written down:

```sh
cmake -S . -B build-sanitize -G Ninja -DVLLM_CPP_BUILD_TESTS=ON \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SANITIZE='address,undefined'
cmake --build build-sanitize --target test_dots3_note_attn -j 6
VT_POOL_BYPASS=1 UBSAN_OPTIONS=print_stacktrace=1 ./build-sanitize/tests/test_dots3_note_attn
```

There is a second, sharper lesson in HOW the finding was attributed. CI named
`test_dots3_note_attn.cpp:4161` — a W4b-2 sliding case that does not run the
indexer at all. It is the wrong case: doctest writes its case headers to
buffered stdout while UBSan writes to unbuffered stderr, so the report lands
under whichever header was flushed last. Running each case alone (with the
pattern truncated at its first comma, because doctest's `-tc` splits on commas)
found the two that really abort — `W4b-2: what the device path STILL refuses`
and `W4b-3c: a RESUMED request past index_topk still refuses BY NAME`. **A
sanitizer report's neighbouring test-case header is not its test case.**

**RED before, GREEN after**, build exit and run exit captured separately, on
`0d4c773dd` + this repair:

| | build | run | result |
|---|---|---|---|
| `test_dots3_note_attn`, sanitizer lane, PRE-fix | exit 0 | **exit 1** | `cpu_layernorm.cpp:33` misaligned load, aborts inside the refusal cases |
| `test_dots3_note_attn`, sanitizer lane, POST-fix | exit 0 | exit 0 | **43 cases / 3,942 assertions**, 0 failed, 0 runtime errors |

**A direct detector, so the next recurrence does not need this fixture.**
`tests/vt/test_ops_layernorm.cpp` gains `layer_norm reads bf16 inputs whose bytes
start at an ODD address`: the same bf16 bytes placed at `blob.data() + 1` for x,
weight and bias, asserted odd, and the output compared BIT-for-BIT against the
aligned control. Its bite is measured, not assumed — reverting `LoadF32At` to the
typed load in a scratch copy and rebuilding gives build exit 0 and **run exit 1**
with the misaligned-address report at `cpu_layernorm.cpp:33`; with the repair the
suite is 6 cases / 9,263 assertions, 0 failed. On a PLAIN build the case passes
either way, and that is stated in the case's own comment rather than left for a
reader to discover: the detector for this class is the sanitizer lane, and a
green plain gate says nothing about it.

**Nothing else moved.** Every gate this brick names was re-run under the SAME
sanitizer lane, all 0 runtime errors and every count equal to its pre-fix value:
`test_ops_dsa_indexer` 10/176, `test_ops_mla_attn` 23/289,324,
`test_mla_attention_block` 13/2,247,730, `test_deepseek_v2_forward` 11/1,052,
`test_dots3_note_scaffold` 26/110,819, `test_deepseek_v4_dsa` 13/38. An
assertion count that moved would have meant the repair changed numerics rather
than only the load width, and none did.

**Four sibling copies of the same `LoadF32At` remain**, in
`src/vt/cpu/cpu_conv1d_depthwise.cpp`, `cpu_conv2d.cpp`, `cpu_conv3d.cpp` and
`cpu_attn_relpos.cpp`. They are the same defect waiting for the same trigger — a
loaded weight or bias reaching them through a borrow — and they are #627's to
close, which is what that issue's "one site UBSan cannot see" and its request to
grep for siblings already asks for. They are listed under `## Owed` rather than
repaired here, because none is reached by this row and repairing an unreached
kernel without a red-before is the shape this protocol refuses.

#### Stop conditions

Stop and report on ENOSPC; on a gate red that cannot be attributed; if the seam
cannot express the indexer additively (return `NEEDS_DECISION` rather than
writing a second path); or if the combined change stops being reviewable in one
pass.

### 4.10 W5 puts the MoE on the decode path, and the released config stops refusing

**Issue [#699](https://github.com/mudler/vllm.cpp/issues/699); the nextn half is
[#2176](https://github.com/mudler/vllm.cpp/issues/2176).** Branch
`row/MODEL-MM-dots3-note-W5`, base `8cf0808253ed49f11cf89799595a7846821d9ac6`.

W4b-3c left `Dots3NoteDeviceRefusal` with exactly two branches: the MoE layer,
which names W5, and the nextn tail, which names W10. This brick removes both and
nothing else, so the RELEASED `dots-studio/dots3-note-prev` config is
representable for the first time. Removing them is two different kinds of work
and the difference is worth stating before either: the MoE branch needs a block
written, and the nextn branch needs a refusal DELETED, because it is stricter
than upstream.

#### The released MoE arm IS bf16, which is unusual on this row

Every other arm this row has met is a deferral. This one is not. From the
committed full index
(`tests/vllm/models/fixtures/dots3_note_prev/index_full.json`, revision
`1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b`, headers only, no tensor byte read):

| Tensor family | dtype | shape | count |
|---|---|---|---:|
| `model.layers.N.mlp.experts.E.{gate,up}_proj.weight` | BF16 | `[1536, 5120]` | 11520 each |
| `model.layers.N.mlp.experts.E.down_proj.weight` | BF16 | `[5120, 1536]` | 11520 |
| `model.layers.N.mlp.shared_experts.{gate,up}_proj.weight` | BF16 | `[1536, 5120]` | 45 each |
| `model.layers.N.mlp.shared_experts.down_proj.weight` | BF16 | `[5120, 1536]` | 45 |
| `model.layers.N.mlp.gate.weight` | BF16 | `[256, 5120]` | 45 |
| `model.layers.N.mlp.gate.e_score_correction_bias` | **F32** | `[256]` | 45 |

45 MoE layers x 256 routed experts = 11520. The bias is the ONLY dtype
exception in the whole MoE block, and it is F32 upstream too
(`deepseek_v2.py:322-324`, `torch.empty(config.n_routed_experts,
dtype=torch.float32)`). So W5's bf16 arm is not a placeholder arm chosen because
the real one is out of reach — it is the arm the release ships, and the loader
this brick writes reads the released bytes.

**The shared expert's intermediate is `moe_intermediate_size *
n_shared_experts`, and the index says so.** `[1536, 5120]`, not the dense
layers' `[13824, 5120]`. A port that read `intermediate_size` there would build
a 13824-wide MLP and fail at load rather than silently — but only if the load
checks the shape BY NAME, which is why W5a does.

**What that arm weighs, measured over the same index.** The routed experts alone
are 543.58 GB of the checkpoint's 576.89 GB (94.23%); with the shared experts
(2.12 GB) and the routers (0.118 GB) the whole MoE block is 545.82 GB, or
**94.62%**. Nothing this project owns holds that in bf16, and the 298.67 GB fp8
sibling does not fit either (§6.2). "Loadable" is therefore not "runnable end to
end" on this row, and W5 does not change that.

#### The upstream delta over `DeepseekV2MoE` is four items, and at TP=1 three of them are zero

Re-derived at `bc2d63e650`, the revision W4b-2 and W4b-3c read.
`git diff bc2d63e650 5559679229 -- vllm/models/dots3_note/` is EMPTY for the
`model.py` this section cites, so these anchors are the pin's and the head's
alike. Each carries a uniqueness discriminator, because several of the obvious
strings are NOT unique in their file and a bare `grep -c` would have lied:

| What | Anchor | `grep -c` of the discriminator |
|---|---|---:|
| the class | `nvidia/model.py:76` `class Dots3NoteMoE(DeepseekV2MoE)` | `class Dots3NoteMoE` = **1** |
| the shared expert lifted OUT of the base | `model.py:87-99` (`routed_config` with `n_shared_experts` set to `None`) + `:101-113` | `self.shared_experts = DeepseekV2MLP` = **1** |
| the block padding | `model.py:63` `def _padded_mlp_size` | `def _padded_mlp_size` = **1** (bare `_padded_mlp_size` = **3**: `:63`, `:103`, `:532`) |
| the unfused add | `model.py:125-127` `super().forward(...) + self.shared_experts(hidden_states)` | `def forward` = **3** (`:115`, `:310`, `:462`) — the discriminator is the FIRST `def forward` after `class Dots3NoteMoE`, at `:115` |
| the TP-only all-reduce | `model.py:100`, `:130-131` | `use_sequence_parallel_moe` = **1**, at `:540`, where the DECODER LAYER sets it `False` |
| the routed scale's destination | `model.py:527` `apply_routed_scale_to_output=False` | `apply_routed_scale_to_output` = **3** (`:85`, `:94`, `:527`); the discriminator is the one inside `if is_moe:` at `:520-528` |
| the base block | `deepseek_v2.py:287` `class DeepseekV2MoE(nn.Module)` | `class DeepseekV2MoE` = **1**; its `forward` is `:406` (`def forward` = **11** in that file) |
| the router formula | `fused_moe/router/grouped_topk_router.py:80` `def grouped_topk` | `def grouped_topk` = **1** (bare `grouped_topk` = **16**) |

The four items, and what each costs us:

1. **The shared expert is lifted out of the base and added unfused.**
   `routed_config = copy.copy(config)` then
   `object.__setattr__(routed_config, "n_shared_experts", None)` (`:88-90`),
   so `DeepseekV2MoE.__init__` takes the `config.n_shared_experts is None`
   branch at `:354-355` and sets `self.shared_experts = None`. The base's
   `forward` is therefore purely routed, and `Dots3NoteMoE.forward` adds
   `self.shared_experts(hidden_states)` itself at `:125-127`. **Numerically
   this is the same function `DeepseekV2MoE` computes when it owns the shared
   expert** — `moe_runner.py`'s `shared_output + fused_output` — which is why
   `vt::MoeCombine`'s optional `shared` term expresses it exactly and why
   `deepseek_v2.cpp`'s `MoeBlock` did not have to change.
2. **`_padded_mlp_size` is the IDENTITY here, twice over.** `:69-70` returns
   `intermediate_size` unchanged when `block_size is None`, and the released
   bf16 checkpoint carries no `quantization_config` at all (verified against
   the committed `config.json`: the key is absent). Even with the fp8 sibling's
   `weight_block_size = [128, 128]`, at `tp_size = 1` the formula reads
   `blocks = (1536 + 127) // 128 = 12`, then `((12 + 1 - 1) // 1) * 128 * 1 =
   1536` — the input. **Do not port it.** There is no code to mutate and the
   mutation table says so rather than showing a green row.
3. **`reduce_results=False` plus `tensor_model_parallel_all_reduce`** (`:100`,
   `:130-131`) is TP-only. At TP=1 the all-reduce is the identity and the
   decoder layer passes `reduce_results=False` anyway (`:524`).
4. **The sequence-parallel path is DEAD.** `Dots3NoteDecoderLayer.__init__` sets
   `self.use_sequence_parallel_moe = False` unconditionally at `:540` — that is
   [vllm#52172](https://github.com/vllm-project/vllm/pull/52172), "Disable
   sequence parallelism for Dots3 NOTE", the change that landed the day before
   this spec's W0. `gather_output` at `:121` is therefore always false and
   `:122-123` and `:128-129` never run.

**The router GEMM stays bf16.** `_get_moe_router_dtype` (`deepseek_v2.py:131`)
returns `torch.float32` only for `model_type == "glm_moe_dsa"` or an explicit
`moe_router_dtype: "float32"`; `dots3_note` is neither, so it returns `None` and
`GateLinear` runs at the model dtype. There is no fp32 router on this model, and
recording that is the `porting.md` memory-format check for this path: a
too-WIDE router is exactly the dtype defect no token gate can see.

**`deepseek_v2.cpp`'s recorded deviation (a) does NOT apply here.** That
deviation exists because vLLM's CUDA path selects
`apply_routed_scale_to_output=True` and applies `routed_scaling_factor` to the
combined routed OUTPUT, while we apply it to the routing WEIGHTS. dots3-note
passes `apply_routed_scale_to_output=False` (`model.py:527`), so upstream puts
the factor inside `grouped_topk` (`grouped_topk_router.py:159-160`) — which is
`MoeRouterTopKArgs::routed_scaling_factor`, the same place we put it. The two
sides agree by construction rather than by an argument about linearity. On the
released config the factor is 1.0 regardless.

#### Writing a model-local MoE block IS the seam, and hoisting DeepSeek's is not

`include/vllm/model_executor/moe_placement_seam.h` states the contract in its
own prose: every architecture has its own block with the shape
`(Dev, weights, params, dh, T) -> DBuf`, and the SEAM is `RunMoePlaced`, which
closes over the architecture's types with a lambda. `Dots3NoteMoeBlock` is
therefore the seam being used as designed, not a parallel path.

Hoisting `deepseek_v2.cpp`'s private `MoeBlock` was considered and REJECTED,
and the reason is recorded so the next MoE model does not re-litigate it. That
function is keyed on `DeepseekV2MoeWeights` and `DeepseekV2Params`; making it
serve dots3-note means either a shared weights interface (which the seam's own
header says it deliberately does not need) or templating it, and either way the
edit lands on the SACRED DeepSeek-V2 path — 8/8 token-exact on DeepSeek-V2-Lite
— to serve a model with no oracle at all. A two-model change against a SACRED
path, to save one function, on a row whose §6.4 says nothing here can be
compared against vLLM, is the wrong trade. The duplication is ~40 lines of `vt`
calls; the risk is a token-exact gate on a different model.

**No `vt` op changes and none is wanted.** `vt::MoeRouterTopK` already accepts
`num_expert_group = 1 / topk_group = 1`; `vt::MoeSiluMul` and `vt::MoeCombine`
are registered on CPU and CUDA both; `MoeCombine`'s optional `shared` term is
exactly upstream's `+ self.shared_experts(x)`. If W5 had needed a second `vt` op
the design would have been wrong, and the instruction was to return
`NEEDS_DECISION` rather than write one.

#### W5a — weights, and a blockwise-fp8 refusal that names the missing part

`Dots3NoteMoeWeights` joins `Dots3NoteDenseMlp` on
`Dots3NoteLayerDeviceWeights`, and `MaterializeDots3NoteDevice` picks one or the
other per layer from `p.is_moe_layer(l)` — which it must, because on a MoE layer
`mlp.gate_proj.weight` does not exist and the current code loads it
unconditionally. Every shape is checked BY NAME through the existing
`RequireShape`, and the shared expert's is asserted as `moe_intermediate_size *
n_shared_experts` so a port that reached for `intermediate_size` refuses loudly.

**The fp8 sibling gets a named refusal, and it is not hypothetical.** Fetched
from the HF API at this brick: `dots-studio/dots3-note-prev-fp8`'s `config.json`
carries `quantization_config = {"quant_method": "fp8", "fmt": "e4m3",
"activation_scheme": "dynamic", "weight_block_size": [128, 128]}`, and its
`model.safetensors.index.json` (73029 entries) ships a `weight_scale_inv` beside
every expert projection — `model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv`
and its 3 x 256 x 45 siblings — while `mlp.gate.weight` has none. At
`[1536, 5120]` with a `[128, 128]` block that scale is `[12, 40]`.
`dense_loaders::MaterializeBf16Source`
(`include/vllm/model_executor/models/dense_weight_loaders.h`) looks up
`<name>_scale`, not `weight_scale_inv`, and accepts only a per-tensor or
per-output-ROW scale (`n_scale == 1 || n_scale == rows`). So today an fp8
dots3-note checkpoint throws `tensor not found: ..._scale` — a bare miss that
names nothing. AGENTS.md requires an arm to refuse naming the missing part, so
the refusal is keyed on the CONFIG (`quantization_config.weight_block_size`)
rather than on a tensor lookup, and it fires before any bf16 loader runs.

**Keying it on the config is also what keeps the per-ROW case safe.** If a
future dots3-note republish shipped a per-output-row `_scale` instead of a
blockwise `weight_scale_inv`, `MaterializeBf16Source` would silently dequantize
it and run a bf16 GEMM on an fp8 checkpoint. That is numerically plausible
output from an arm nobody ported, and no token gate on this row could see it
(§6.4 — there is no token gate). The config-keyed refusal turns that into a
message. The blockwise-fp8 MoE itself is **W9**.

#### W5b — the block, the forward, and one thing recorded against W9

`Dots3NoteMoeBlock(Dev, const Dots3NoteMoeWeights&, const Dots3NoteParams&,
const Tensor& dh, int64_t T) -> DBuf`, routed through `vllm::RunMoePlaced` from
the layer loop, with `layers::UnquantizedMlpGateUpMethod` for the shared expert
so the mergeable-MLP seam carries it exactly as the dense layers' MLP is
carried. The unconditional `DenseMlp(...)` call in `ForwardDevice` becomes a
branch on the layer's kind, and the MoE branch of `Dots3NoteDeviceRefusal` goes.

**Owed, recorded rather than built:** `vt::QuantFp8Group` has no `use_ue8m0`
rounding. It does not bite at W5, because it is the ACTIVATION quantizer and W5
is entirely on the bf16 path — nothing in this brick calls it. It probably does
bite at **W9**, because upstream's blockwise-fp8 MoE routes through DeepGEMM
with e8m0 scales, and a port that quantizes activations with plain
power-of-two-free scaling there will disagree with the kernel upstream runs.
Recorded here against W9 with the reason so it is not re-derived from scratch.

**GGUF k-quants stay owed and refused by name** at `dots3_note_registry.cpp`.
W5 does not weaken that refusal and does not touch it.

#### W5c — the nextn refusal is STRICTER than upstream, and that is the defect

Issue [#2176](https://github.com/mudler/vllm.cpp/issues/2176). This is a
mirror-fidelity fix, not a feature: vLLM does not refuse a checkpoint that
ships nextn weights, it DROPS them from the main model. Three anchors, at
`bc2d63e650`, each with its discriminator:

| Where | Line | `grep -c` |
|---|---|---:|
| `vllm/model_executor/models/utils.py` | `:542` `def get_spec_layer_idx_from_weight_name`, matching `model.layers.{base+i}.` or `layers.{base+i}.` at `:559` | `def get_spec_layer_idx_from_weight_name` = **1** |
| `vllm/model_executor/models/deepseek_v2.py` | `:1618-1620` `spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)` / `if spec_layer is not None:` / `continue  # skip spec decode layers for main model` | `spec_layer = get_spec_layer_idx_from_weight_name` = **1** |
| `vllm/models/dots3_note/nvidia/model.py` | `:624` `if name.startswith("mtp."):` / `continue`, inside `Dots3NoteModel._adapt_weights` (`:619`), reached from its `load_weights` (`:677-678`) | `startswith("mtp.")` = **1** |

`Dots3NoteLanguageModelForCausalLM` (`model.py:681`) subclasses
`DeepseekV32ForCausalLM`, so the `deepseek_v2.py` skip is on the path this
architecture actually loads through. `num_nextn_predict_layers` is absent from
the released `config.json` and §4 trap 3 correctly defaults it to 1, so every
released checkpoint trips a refusal upstream does not have.

**The repair is the classifier-deferral shape the towers already use**, with one
difference that is forced by the data rather than chosen. `Dots3NoteDeferredTowers()`
keys on a STATIC prefix, and the nextn tail's prefix is config-derived
(`model.layers.{num_hidden_layers + i}.`, plus `model.mtp.`), so it gets a
predicate — `Dots3NoteIsNextnTensor(params, name)` — rather than a table row.
The counter is the point either way: `Dots3NoteAccounting` grows a `nextn`
bucket, and over the released index the split becomes **35362 language / 19
nextn / 2195 vision / 430 audio = 38006**, against W2's 35381 / 2195 / 430. The
19 are `model.layers.46.*` (18) and `model.mtp.embed_tokens.weight`, and the
committed fixture's `bucket_totals` is updated to the same split with a note
that 35362 + 19 is W2's 35381.

**The nextn names stay ENUMERATED**, so `Dots3NoteAccounting::missing` still
refuses a checkpoint that claims a nextn layer and does not ship it. Only the
BUCKET moves. That is the smaller change and it keeps a check W2 earned.

Then the headline: `Dots3NoteDeviceRefusal(released_params).empty()`. That is a
`true -> false` flip on an assertion that exists today, and it is what this
brick is for.

#### The gate design — a discrete assertion, because the hazard here is discrete

No oracle (§6.4 option B). The instrument is an INDEPENDENT double-precision
reference of the MoE block, transcribed from `grouped_topk_router.py:80`,
`deepseek_v2.py:406` and `model.py:115`, and **written without reading
`src/vt/cpu/cpu_ops.cpp`** — a reference that shares a helper with the code it
gates measures consistency with itself.

**Why a single continuous tolerance says almost nothing about this block.**
Router logits are stored bf16, whose relative ulp is `2^-8 = 3.906e-3`. Through
`sigmoid' <= 0.25` that is a score perturbation of order `1e-3` to `5e-3`. With
`E = 256` and sigmoid scores spread over `[0, 1]`, the typical gap between the
8th and 9th order statistics is about `1/256 = 3.9e-3` — the SAME order. A
fixture whose router logits are uniform noise is therefore a coin flip on
whether the selected set matches, and a relative-error bound would absorb the
difference between "the same 8 experts, rounded" and "a different expert
entirely". §4.9 records this exact defect one brick ago on the DSA indexer: a
7.43e-4 selection margin against a 1.28e-3 ulp, and the repair was the FIXTURE,
never the threshold.

So the gate carries five things, and the first three are the ones a continuous
bound cannot give:

1. **Selection-set equality**, asserted as a SET. `torch.topk(..., sorted=False)`
   leaves the order unspecified (`use_sorted = envs.VLLM_BATCH_INVARIANT`,
   `grouped_topk_router.py:134`) and `vt::MoeCombine` sums over the slots, so
   order is not part of the contract and asserting it would pin something
   upstream does not promise.
2. **The minimum decision margin**, PRINTED, and required to exceed **4x** the
   bf16 score ulp at the fixture's own scale — the same bar W4b-3c stated and
   met at 48.8x. The margin is the biased-score gap across the k-th boundary:
   `min over tokens of (score+bias)[k-th selected] - (score+bias)[best rejected]`.
3. **The number of DISTINCT experts activated across the batch**, PRINTED. A
   fixture in which every token picks the same k experts has not tested routing,
   and this brick's whole risk is which experts get picked.
4. **A deliberate exact tie, OFF the boundary.** Two experts are given
   byte-identical router-gate rows and identical bias, so their biased scores
   tie EXACTLY in double for every token, and both sit inside the selected set.
   The set is then unambiguous whatever the tie rule is, and the assertion is
   that an exact tie changes nothing — which is the claim upstream's
   `sorted=False` actually supports. A tie ON the k-th boundary is deliberately
   NOT in the fixture: the selected set would be genuinely ambiguous, upstream
   does not specify which side wins, and gating it would pin our kernel's
   accident as a contract.
5. **A designed, bias-dominated fixture.** The bias values separate an
   always-selected tier from an always-rejected one by far more than the score
   spread, while a middle tier at equal bias lets the LOGITS decide the last
   slot per token — so the selection is genuinely token-dependent and a port
   that dropped the bias picks a different pool. The always-selected experts
   carry DIFFERENT biases from the contended one, which is what makes the
   nearest mechanism visible.

**The continuous bound follows W4b-2's shape**: a residue near 0.02-0.03, a
bound of **0.06**, and a nearest mechanism at **>= 0.15**, the bound sitting
near the geometric mean. The nearest mechanism, and the single most likely port
defect, is **the correction bias applied to the routing WEIGHT as well as to the
selection** — upstream is explicit that it is not
(`grouped_topk_router.py:121-123`: "We use biased scores for expert selection
but original scores for routing weights"). The fixture is tuned against that
one. **If a mechanism lands under the residue, the FIXTURE is retuned and the
bound is not.**

**Predicted-GREEN mutations, named in advance rather than discovered:**

- **Every group-stage mutation**, because `n_group = 1 / topk_group = 1` makes
  the group mask all-ones and the stage definitionally inert. The standing
  coverage is the UNGROUPED-ONLY REFUSAL in `ParseDots3NoteParams`
  (`dots3_note.cpp`, the `n_group == 1 && topk_group == 1` check), and W5's gate
  RE-ASSERTS it rather than inheriting it silently. Note what is NOT inert:
  setting `MoeRouterTopKArgs::num_expert_group` to 0 selects the pre-W3
  ungrouped SOFTMAX path verbatim, which ignores both `scoring_func` and the
  bias, so that mutation is RED and is in the red table.
- **`routed_scaling_factor`**, which is 1.0 on this config, so replacing
  `p.routed_scaling_factor` with a literal `1.0f` is inert.
- **`_padded_mlp_size`**, identity at TP=1 on this checkpoint and deliberately
  not ported, so there is no line to mutate. Recorded as an absence rather than
  shown as a green row.

Every mutation is driven through the committed `scripts/mutation-harness.py`,
anchors are asserted UNIQUE before they are mutated, and the COMPILER EXIT is
printed beside every row. A mutation that does not build is `NOT A RESULT`, and
a non-building mutation reads as a passing test if nobody looks.

#### Reachability

The chain is `ModelRegistry::Resolve` -> `LoadDots3NoteForCausalLM` over a real
`SafetensorsFile` -> `MaterializeDots3NoteDevice` -> `ModelRegistry::Forward` ->
the layer loop -> `vllm::RunMoePlaced` -> `Dots3NoteMoeBlock`. The smallest
failing test enters at `ModelRegistry::Resolve`, exactly as W4a, W4b-2 and
W4b-3c do; no case constructs `Dots3NoteMoeWeights` by hand. Three deletion
mutations are owed and are in the table: the block call in the layer loop, the
MoE arm of materialization, and a restored MoE branch of the refusal.

#### Device

The correctness gate is CPU-only and needs no lease. ONE device run is owed and
taken: `vt::MoeGroupedGemmBf16` and `vt::MoeGroupedGemmBf16GateUpSilu` are
CUDA-only, so `Dots3NoteGroupedMoeEligible` selects the reference arm on CPU and
the grouped arm on CUDA, and dots3-note's ROUTING to them plus the resident
expert-pointer upload is new code that the CPU gate never executes. The proof is
an on-device assertion COUNT against a `CUDA_VISIBLE_DEVICES=""` control on the
SAME binary: a doctest case that returns before its first assertion scores
PASSED with zero assertions, so the delta is the evidence and the verdict is
not. The path needs no FA-2, no fp8 and no NVFP4, so all three fleet devices
qualify; `orin:gpu0` is preferred because §4.8 proved the recipe on this row.
`rc run` only, never `ssh`, bounded with `--max-runtime`.

#### The gate, met

Measured at `9a500f3e1e8bc2f034cce1fdac45a3c632446af1` on the devbox
(`mudler-ubuntu-box`, x86_64, gcc 13, `CMAKE_BUILD_TYPE=Debug`,
`VLLM_CPP_BUILD_TESTS=ON`, CUDA OFF). **Build exit and run exit are separate
numbers, because a non-building mutation reads as a passing test when nobody
looks.**

| binary | build | cases | assertions | run exit |
|---|---|---|---:|---|
| `test_dots3_note_attn` | `BUILD_RC=0` | 51 / 0 failed | 6,888 / 0 failed | `RC=0` |
| `test_dots3_note_scaffold` | `BUILD_RC=0` | 26 / 0 failed | 110,832 / 0 failed | `RC=0` |
| `test_model_registry` | `BUILD_RC=0` | 24 / 0 failed, 1 skipped | 975 / 0 failed | `RC=0` |

**The continuous half.**

| quantity | value | against |
|---|---:|---|
| residue (device bf16 against the double reference) | **0.011902** | 0.198x the bound |
| the bound `kMoeRel` | **0.06** | 5.04x the residue |
| nearest mechanism | **0.207964** | 3.47x the bound |

The bound sits just above the geometric mean of the residue and the nearest
mechanism (0.0497), which is W4b-2's shape and the reason it is not a hugged
threshold.

**The DISCRETE half, which is the part a bound cannot give.**

| quantity | value | bar |
|---|---:|---|
| minimum decision margin | **0.0633073** | — |
| bf16 score ulp at this fixture's scale (`0.25 * max\|logit\| * 2^-8`, `max\|logit\|` 3.76543) | 0.00367718 | — |
| margin / ulp | **17.2x** | > 4x |
| distinct experts activated across the batch | **6 of 8** | >= 4 |
| the deliberate exact tie, `max\|biased[0] - biased[1]\|` | **0.0** | exactly 0 |
| selection-SET equality against the reference | **14 of 14** (layer, token) decisions | all |
| the group stage deleted from the reference | **0.0 absolute** | exactly 0 |

**Every mechanism, printed by the case from the numbers it just measured.**

| mechanism mutated in the REFERENCE | relative | x the bound | x the residue |
|---|---:|---:|---:|
| `norm_topk_prob` dropped (`:156-157`) | 0.791637 | 13.19 | 66.51 |
| `routed_scaling_factor` 1.7 (this config's is 1.0) | 0.580572 | 9.68 | 48.78 |
| `top_k - 1` | 0.478647 | 7.98 | 40.22 |
| the bias applied to the routing WEIGHT too | 0.405859 | 6.76 | 34.10 |
| the shared expert dropped (`model.py:127`) | 0.358613 | 5.98 | 30.13 |
| the correction bias dropped from the SELECTION | 0.319568 | 5.33 | 26.85 |
| `top_k + 1` | 0.314643 | 5.24 | 26.44 |
| **softmax scoring instead of sigmoid (`:112-117`)** | **0.207964** | **3.47** | **17.47** |

**THE BRIEF'S PREDICTION ABOUT WHICH MECHANISM IS NEAREST WAS WRONG, and the
first fixture would have shipped a hole because of it.** The design section
above names the bias-in-the-routing-weight defect as the nearest mechanism and
says to tune against that one. It is not nearest: at this fixture it reads
0.4059, sixth of eight. The nearest is SOFTMAX-versus-SIGMOID scoring, and on
the FIRST fixture it read **0.0400 — below the 0.06 bound**, so a port that
wrote `kSoftmax` into the router args would have passed a green gate. The
mechanism is small precisely where the fixture was designed to be safe: with
small router logits, sigmoid is near-linear and the two scoring functions barely
separate after renormalisation.

**The repair was the fixture, twice, and never the bound.** The two knobs and
what each measured:

| `contended_gain` (experts 2-5's `down_proj` amplitude) | bias-in-weight |
|---:|---:|
| 1 | 0.0703 - 0.2262 across six seeds |
| 3 | 0.2620 - 0.7127 across the same six |

| `router_amp` | margin / ulp | distinct | residue | softmax arm |
|---:|---:|---:|---:|---:|
| 0.09 | 19.4x | 6 | 0.0230 | **0.0400** |
| 0.18 | 19.3x | 6 | 0.0193 | 0.0770 |
| 0.30 | 18.7x | 6 | 0.0190 | 0.1126 |
| 0.45 | 18.1x | 6 | 0.0162 | 0.1515 |
| **0.60** | **17.2x** | **6** | **0.0119** | **0.2080** |

0.45 is the smallest value that clears the 0.15 fixture-quality floor, and it
clears it by one percent. A guard met by one percent is the hugged threshold
this project keeps naming, so the fixture takes 0.60.

The seed was chosen the same way, over six seeds x {gain 1, 3} x {shared
amplitude 0.5, 0.15}, and the spread is the argument for measuring rather than
assuming: the minimum decision margin ranged **1.51x to 25.8x the ulp** across
that grid. One seed in twelve would have shipped a fixture whose selection is a
coin flip, which is exactly §4.9's DSA-indexer defect one brick later.

#### The mutation table

Driven through the committed `scripts/mutation-harness.py`, which refuses a
dirty tree, refuses an absent anchor, prints the diffstat, prints the compiler
error count, and restores the tree byte-for-byte verified by sha256. Every
anchor was asserted UNIQUE (`grep -cF` == 1) before the run. **The compiler exit
is printed beside every row**, because a non-building mutation is `NOT A RESULT`
and reads as a passing test.

Baseline for the `test_dots3_note_attn` rows: `exit=0 cases=51 (0 failed)
assertions=6888 (0 failed)`. Baseline for the `test_dots3_note_scaffold` rows:
`exit=0 cases=26 (0 failed) assertions=110832 (0 failed)`.

| # | mutation | file | built | cc errors | run exit | cases / asserts failed | verdict |
|---|---|---|---|---:|---:|---|---|
| M1 | the router bias passed as `nullptr` | device | YES | 0 | 1 | 2 / 3 | DETECTED |
| M2 | `args.renormalize = false` | device | YES | 0 | 1 | 2 / 4 | DETECTED |
| M3 | `top_k - 1` | device | YES | 0 | 1 | 3 / 0 | DETECTED |
| M4 | the shared term passed as `nullptr` to `MoeCombine` | device | YES | 0 | 1 | 2 / 4 | DETECTED |
| M5 | `args.num_expert_group = 0` (the pre-W3 softmax path) | device | YES | 0 | 1 | 3 / 0 | DETECTED |
| M6 | the shared expert built at `intermediate_size` | device | YES | 0 | 1 | 3 / 0 | DETECTED |
| M7 | the router loaded raw-NK instead of transposed | device | YES | 0 | 1 | 4 / 1 | DETECTED |
| M8 | expert `gate_proj` loaded from `up_proj` | device | YES | 0 | 1 | 2 / 2 | DETECTED |
| M9 | REACH: the layer-loop `RunMoePlaced` call deleted | device | **NO** | **1** | — | — | **BUILD_FAILED — NOT A RESULT** |
| M9r | the same, with `(void)&Dots3NoteMoeBlock;` to satisfy `-Werror=unused-function` | device | YES | 0 | 1 | 3 / 0 | DETECTED |
| M10 | REACH: `lw.is_moe = false` in materialization | device | YES | 0 | 1 | 4 / 1 | DETECTED |
| M11 | REACH: the MoE refusal branch restored | device | YES | 0 | 1 | 10 / 14 | DETECTED |
| M12 | the nextn refusal branch restored | device | YES | 0 | 1 | 5 / 9 | DETECTED |
| M13 | `Dots3NoteIsNextnTensor` never matches `model.mtp.` | model | YES | 0 | 1 | 2 / 2 | DETECTED |
| M15 | the blockwise-fp8 refusal disabled | device | YES | 0 | 1 | 1 / 4 | DETECTED |
| M16 | `weight_block_size` parsed but not stored | model | YES | 0 | 1 | 1 / 1 | DETECTED |
| M14 | the nextn accounting bucket disabled | model | YES | 0 | **0** | 0 / 0 | **SURVIVED on `test_dots3_note_attn`** |
| G1 | `routed_scaling_factor` replaced by the literal `1.0f` | device | YES | 0 | 0 | 0 / 0 | **PREDICTED GREEN — SURVIVED** |
| G2 | `topk_group` replaced by the literal `1` | device | YES | 0 | 0 | 0 / 0 | **PREDICTED GREEN — SURVIVED** |

**M9 is recorded as a build failure rather than dropped**, because it is the
class the harness exists to catch: it failed `-Werror=unused-function` — with
the only call gone, `Dots3NoteMoeBlock` has no user — and a harness that scored
the previous binary's run would have called that a survivor. Roughly one
mutation in five dies this way and this one did; M9r is the same deletion with
the symbol's address taken so the compile stands.

**M14 SURVIVED on `test_dots3_note_attn` and that is an instrument fact, not a
coverage hole** — the nextn accounting bucket is asserted in
`test_dots3_note_scaffold`, which the attention binary does not contain, so the
first run pointed the instrument at the wrong binary. Re-run against the binary
that carries the assertion, with `test_dots3_note_scaffold`'s own baseline of
`exit=0 cases=26 (0 failed) assertions=110832 (0 failed)`:

| # | mutation | file | built | cc errors | run exit | cases / asserts failed | verdict |
|---|---|---|---|---:|---:|---|---|
| M13s | `Dots3NoteIsNextnTensor` never matches `model.mtp.` | model | YES | 0 | 1 | 3 / 6 | DETECTED |
| M14s | the nextn accounting bucket disabled | model | YES | 0 | 1 | 3 / 6 | **DETECTED** |
| M19s | REACH: `lw.is_moe = false` in materialization | device | YES | 0 | 0 | 0 / 0 | SURVIVED — WRONG BINARY AGAIN |

**M19s is the same shape and is recorded rather than hidden.** It is M10's
mutation pointed at the scaffold binary, which carries no MoE forward gate, so a
survival there says nothing. M10 is the same edit against
`test_dots3_note_attn`, which does carry it, and M10 is RED. The row is kept so
a reader can see that two of the twenty-one rows measured a binary rather than a
guarantee, and that both were re-aimed rather than argued away.

**A misleading diffstat in the scaffold rows, and it is THIS SESSION'S error
rather than the harness's.** Every scaffold row's `diff --stat` column names
four paths — this spec and the two `docs/` projections beside the mutated source
— which reads as though the harness mutated four files. It did not.
`mutation-harness.py::require_clean` runs `git status --porcelain` ONCE at
start-up and refuses a dirty tree, and the tree was clean when that run started;
the three record-commit files were edited by this session WHILE the run was in
flight, and `diffstat()` re-reads the whole tree per mutation. The claim was
almost written up the other way round — as a harness weakness — and reading
`require_clean` before recording it is what stopped that. The per-file sha256
restore is unaffected and was verified afterwards: `git diff -- src tests` is
empty at the end of the run. The lesson for the next run is the ordinary one:
do not touch the tree while a mutation pass owns it, even in files the plan
does not name.

**The two PREDICTED-GREEN rows were named before the run, in the design section
above, and both came back green for the stated reason.** `routed_scaling_factor`
is 1.0 on this config and on the released one, so the multiply is the identity.
`topk_group` is 1, so writing the literal is writing the value. Note what is NOT
inert and is therefore in the red table: `num_expert_group = 0` selects
`vt::MoeRouterTopK`'s pre-W3 ungrouped SOFTMAX path verbatim, which ignores both
`scoring_func` and the correction bias, so M5 is red rather than green.
`_padded_mlp_size` has no row at all — it is deliberately not ported, so there
is no line to mutate, and the absence is recorded rather than shown as a green.

#### The residency defect the fresh review found, and what a mutation CAN say about it

W5 shipped `Dots3NoteMoePtrsFor` as a process-lifetime `static std::map<const
Dots3NoteMoeWeights*, Dots3NoteMoePtrs>`, the pre-#237 shape, and cited #237's
own repair as its warrant (review F1,
[#2193](https://github.com/mudler/vllm.cpp/issues/2193)). It now builds into a
`ResidentSlot` the weights own. Two mutations, run through the same harness on
the repaired head, and the pair is what makes the claim honest rather than the
first row alone:

| # | mutation | test | built | cc errors | run exit | cases / asserts failed | verdict |
|---|---|---|---|---:|---:|---|---|
| F1-a | the accessor keys on the WEIGHT'S ADDRESS again | `test_moe_resident_lifetime` | YES | 0 | 0 | 0 / 0 | **SURVIVED** |
| F1-b | the same edit | `test_dots3_note_attn` | YES | 0 | 0 | 0 / 0 | **SURVIVED** |
| F1-c | CONTROL: `static inline ResidentSlot resident_moe` — one slot shared by every block | `test_moe_resident_lifetime` | YES | 0 | 1 | 2 / 3 | DETECTED |
| RV-C | DELETE the `resident_moe` member from `Dots3NoteMoeWeights` | `test_moe_resident_lifetime` | **NO** | 13 | — | — | **BUILD_FAILED** |

**F1-a and F1-b survived, and that is a property of the arm rather than a weak
gate.** `Dots3NoteMoePtrsFor` is file-local to `dots3_note_device.cpp` and is
called only from inside `Dots3NoteGroupedMoeEligible`, which requires a NATIVE
`vt::OpId::kMoeGroupedGemmBf16`. That op is registered for CUDA only
(`src/vt/cuda/cuda_matmul_nvfp4.cu:2722`) and has no CPU reference tier, so no
CPU gate can call the accessor at all. Reverting its body is unobservable here
by construction, and it rides the device run the grouped arm already owes under
`## Owed`.

**F1-c is the positive control, and it is why the two survivals are readable.**
Without it, `SURVIVED` is indistinguishable from a dead harness or an assertion
that never ran. The control mutates the ONE property the CPU cases do pin —
that residency is owned per BLOCK — by making the slot shared, and the suite
reds at 2 cases / 3 assertions with exit 1. So the instrument is armed, the
cases discriminate, and what they cannot reach is named rather than implied.

**RV-C is why F1-a's survival is a statement about the ARM and not about the
member.** F1-a and F1-b revert the accessor's BODY, and nothing on a CPU tier
can call it, so they say nothing either way about whether the accessor and the
member are actually coupled — a reader has to take that from four lines of
source. RV-C removes the member instead, and the compiler answers: **the build
FAILS with 13 `: error:` lines**, so the coupling is machine-checked rather than
read. Measured through the same harness at this head
(`scripts/mutation-harness.py --test test_moe_resident_lifetime`), which refuses
a dirty tree and reports the build status beside the row, and reproduced
directly to see the whole build output rather than the harness's 1200-character
tail.

**The 13 do not fall where the deferral note predicted, and the split is the
interesting part.** Three are inside `Dots3NoteMoePtrsFor`
(`dots3_note_device.cpp:540`, `:541`, `:543`), each reading `has no member named
'resident_moe'`. The other **ten** are in `test_moe_resident_lifetime.cpp`
(`:150`, `:155`, `:156`, `:157`, `:164`, `:165`, `:171`, `:172`, `:184`, `:185`).
That the test file supplies the majority is worth stating plainly rather than
rounding away: the four residency cases bind to the MEMBER by name, so they
cannot be satisfied by a copy of the state kept somewhere else, which is exactly
the property F1-a cannot reach. A BUILD_FAILED is `NOT A RESULT` for a run-exit
question and is a result for this one, because the question RV-C asks is whether
the reference exists at all.

The tree was restored byte-for-byte afterwards — `sha256` of
`dots3_note.h` back to `33b5b2f0e4`, `git status --porcelain` empty — and
`test_moe_resident_lifetime` rebuilt and re-ran at 10 cases / 28 assertions,
exit 0, which is the control proving the restore rather than an assumption that
it worked.

#### Reachability

Three deletion mutations, all red, all through
`ModelRegistry::Resolve` -> `LoadDots3NoteForCausalLM` over a real
`SafetensorsFile` -> `MaterializeDots3NoteDevice` -> `ModelRegistry::Forward` ->
the layer loop -> `vllm::RunMoePlaced` -> `Dots3NoteMoeBlock`:

- **M9r** deletes the production CALL in the layer loop. RED.
- **M10** deletes the MoE arm of materialization (`lw.is_moe = false`). RED.
- **M11** restores the MoE branch of `Dots3NoteDeviceRefusal`, so the forward
  turns the config away before the block runs. RED at 10 cases and 14
  assertions, the widest of the three.

No W5 case constructs `Dots3NoteMoeWeights` or calls `Dots3NoteMoeBlock` by
hand. The one case that calls a `vt` op directly — the selection-set probe on
`vt::MoeRouterTopK` — says in its own body that it is a supplementary DISCRETE
probe and not the reachability-carrying gate.

#### The device run, and the ZERO DELTA it returned

`rc run -d orin:gpu0`, job **`b4b2a08b-35b4-4f54-806a-aa9f3cc3ca37`**, 2026-08-28,
`--max-runtime 180m --idle-timeout 25m`, about 30 minutes of device time.
`rc run` only; the box was never reached by `ssh`. The SHA was PROVEN rather
than asserted: the job cloned in the container and refused to build unless
`git rev-parse HEAD` equalled `9a500f3e1e8bc2f034cce1fdac45a3c632446af1` with
`git status --porcelain` empty. `orin`'s `/workspace` is LOCAL disk and is not
the NAS, so the source came from a `git clone` of this branch rather than from a
staged tarball.

**The toolchain gate did its job, and it is the reason this section can say
"executed" about anything at all.** CUDA 13.0 installs on this box and compiles
for `sm_87`, and it cannot LAUNCH: the smoke program read `SMOKE=NO_DEVICE`
(`SMOKE13_RC=2`), which is `cudaGetDeviceCount` failing against the 540.4.0
driver. The job then installed `cuda-toolkit-12-6` and re-ran the SAME smoke
program, which compiled and launched (`SMOKE126_RC=0`). A job that had gated on
`command -v nvcc` would have reported a compile as an execution.

**What was proven.** The tree CONFIGURES and BUILDS clean for `sm_87` with
`VLLM_CPP_CUDA=ON`: `CONFIGURE_RC=0`, `BUILD_RC=0`, **zero** `: error:` lines
across 547 targets including every `*.cu` object. W5 adds no `-Werror` breakage
on the CUDA tier, which the CPU-only devbox cannot show.

**What was NOT proven, and this is the result rather than a caveat.** The
per-case assertion counts, device against a `CUDA_VISIBLE_DEVICES=""` control on
the SAME binary:

| case | device | control | delta |
|---|---:|---:|---:|
| the MoE layer through `ModelRegistry::Forward` | 295 | 295 | **0** |
| every MoE mechanism past the bound | 283 | 283 | **0** |
| the SELECTION set-equal to the reference | 114 | 114 | **0** |
| the mixed dense+MoE forward is DETERMINISTIC | 468 | 468 | **0** |

Whole binary, both ways: 51 cases / 6,888 assertions / `SUCCESS!`, `DEVICE_RC=0`
and `CONTROL_RC=0`.

**A zero delta means the device did not participate, and the cause is in the
FIXTURE rather than in the model.** Every dots3-note model case in this file
builds its queue as `vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}` —
`grep -c kCUDA tests/vllm/models/test_dots3_note_attn.cpp` is **0** — so
`ModelRegistry::Forward` runs on a CPU queue whatever the box has, and
`Dots3NoteGroupedMoeEligible` is false because
`vt::OpRegistered(kMoeGroupedGemmBf16, kCPU)` is false. The reference arm ran on
both sides, identically, which is exactly what identical counts say.

**So the grouped arm of `Dots3NoteMoeBlock` has NO execution evidence, and it is
recorded as owed rather than claimed.** It is not DEAD code — production reaches
it whenever the engine's queue is CUDA, which is the ordinary configuration on a
CUDA build — it is UNGATED. Closing it needs a device-queue variant of the
bench: a `kCUDA` `vt::Queue`, a `PagedKvCache` in device memory rather than
`w4a::MlaCachePool`'s host vectors, and the resident-weight uploads that follow.
That is a test-infrastructure change of its own size, and W4b-2 and W4b-3c hit
the same wall from the other side — §4.8 says their CUDA evidence is
"KERNEL-level parity on two ops. It is not the end-to-end model gate", and no
brick on this row has yet run the MODEL on a GPU.

**The instrument is what makes this reportable at all.** Had the job printed
only `SUCCESS!` and the exit code, the run would have read as a device
execution: the verdict is identical on both sides. The assertion COUNT is the
only column that distinguishes them, and it distinguishes them by being the
same.

#### Risks

- **R-W5-1 — the fixture's selection margin collapses under retuning.** The
  mitigation is the printed margin and the 4x bar; if it cannot be met by
  fixture design the brick STOPS and reports rather than widening the bar.
- **R-W5-2 — the reference agrees because it shares a helper.** The mitigation
  is the rule above: the reference is written from the upstream Python, and
  `cpu_ops.cpp` is not read while writing it.
- **R-W5-3 — the released config becoming representable reads as "runnable".**
  It is not. The MoE is 94.62% of a 576.89 GB checkpoint. Every claim this
  brick makes is bounded by that and says so.

#### Stop conditions

Stop and report on ENOSPC. Stop if the continuous bound cannot be met without
widening it. Stop if the nearest mechanism cannot be pushed above the residue by
fixture design. Stop if W5c needs more than the classifier-deferral shape. Stop
and return `NEEDS_DECISION` if the seam needs a new `vt` op.

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
| `dots3-note-prev-fp8` | 298.67 GB (298,673,280,504 B) | no | no |
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
   298.67 GB FP8 checkpoint is more than twice the box's 122 GB of memory, so the
   model cannot be resident whatever the disk holds. **The disk half of this
   argument is WITHDRAWN.** W0 wrote it as "290 GB exceeds both its 122 GB of
   RAM and its 123 GiB of free disk — the checkpoint will not even land", and
   the very next measurement of the same volume read 362 GB free, which is more
   than 298.67. On that number the checkpoint lands and then fails to load. Both
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
non-GB10 CUDA host), our own low-bit arm end to end, and every unit/brick gate
in §7 THAT IS NOT ON THE FA-2 PATH — none of which need the 298.67 GB checkpoint.

**The FA-2-gated tests are the exception, and this designation never reached
them.** Thor's sm_110 is outside `VT_CUDA_FEATURE_TABLE`'s `fa2` row, so
`VT_FA2_ARCHS` resolves EMPTY there and `MlaPrefillAttentionCuda` throws instead
of computing. `test_ops_mla_prefill` and `test_mla_attention_block`, this row's
own two CUDA gates, are two of the FOUR names §7's W0.5 failure table already
records red on Thor under the cause "no vendored FA-2". This designation is
therefore a design error on that path rather than a stale reading. §4.8 carries
the derivation and names `orin:gpu0` (8.7) and `dgx:gpu0` (12.1a) as the hosts
that CAN serve it. Pick the CUDA host by CAPABILITY, not by availability.

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
  option B. **That consequence is stronger than "cannot be verified end to
  end", and this section is where the designation above must be read against
  it.** `test_ops_mla_prefill` and `test_mla_attention_block` are two of the
  four names in the failure table just above, under the cause "no vendored
  FA-2". Thor's sm_110 is outside `VT_CUDA_FEATURE_TABLE`'s `fa2` row
  (`cmake/CudaArchFeatures.cmake`), so `VT_FA2_ARCHS` resolves empty,
  `VLLM_CPP_FLASH_ATTN` is never defined, and `MlaPrefillAttentionCuda` throws
  (`src/vt/cuda/cuda_mla_prefill.cu:179-183`). **Thor therefore cannot gate ANY
  windowed MLA prefill on this row, at any point, and the header's designation
  does not reach that path.** The fleet devices that can are `orin:gpu0` (8.7)
  and `dgx:gpu0` (12.1a); `orin:gpu0` discharged W4b-2's CUDA half on
  2026-08-26 (§4.8).

  And W9's blockwise-FP8 arm still has no native kernel on this box —
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
- **W4 — the attention bricks. SPLIT into W4a and W4b**, which is what the
  bullet this replaces predicted ("the largest brick; likely splits further").
  The split line is the GEOMETRY: W4a is the full-attention layer, which shares
  the paged MLA cache DeepSeek already runs on; W4b is everything the SLIDING
  geometry needs, which is new machinery on both the metadata and the kernel
  side (§2.3).
- **W4a — the full-attention layer ON THE DECODE PATH. DONE**
  (`row/MODEL-MM-dots3-note-W4a`,
  [#699](https://github.com/mudler/vllm.cpp/issues/699)). Exactly the two items
  W3 left under `## Owed`: `mla::ForwardMlaAttentionBlock` grew four optional
  fields (two `double` scales on `MlaBlockDims`, one norm weight and one gate
  weight on `MlaBlockWeights`), and `Dots3NoteModel::ForwardDevice` became a
  real forward for a config whose every layer is `full_attention` with a DENSE
  MLP, reached through `ModelRegistry::Forward`. **The SACRED DeepSeek-V2 path
  is byte-identical, measured before/after over raw output bytes on SIX
  geometries spanning the seam's whole branch space, with the base arm built in
  its own tree.** Gate met: `test_dots3_note_attn`, **18 cases / 638
  assertions**, against W3's independent double reference lifted to the whole
  model; `test_mla_attention_block` **12 / 2247715**. §4.6 carries the evidence,
  the 18-row mutation table and the three things it measured rather than
  assumed. **The released checkpoint still refuses**, at layer 1 (MoE) and
  layer 2 (sliding).
- **W4b — sliding-window MLA. SPLIT into W4b-1 and W4b-2**, on the same line
  W3/W4a used: the maths first, the decode path second. The bullet this replaces
  scoped the whole §2.3 stack PLUS the three refusals W4a handed on, and §4.7
  measures why that is two bricks rather than one — a padded cache row is a
  change inside `vt::ConcatAndCacheMla` / `vt::MlaDecodeAttention` / the MLA
  prefill gather on BOTH backends, so it carries a CUDA half no CPU-only box can
  verify and a byte-identity obligation on the SACRED DeepSeek-V2 path.
- **W4b-1 — the sliding maths and the §2.3 machinery, as host code. DONE**
  (`row/MODEL-MM-dots3-note-W4b-1`,
  [#699](https://github.com/mudler/vllm.cpp/issues/699)). `SlidingAttnDims`
  resolved from the released config, `SwaGatherLen`, `GatherSwaKv`,
  `ApplySwaScoreMask`, `BuildSlidingWindowMetadata`, the
  `PaddedMlaCacheSpec`/`_logical_cache` pair, and `ForwardSlidingAttention` —
  which computes the attention the way upstream does, the ABSORBED MQA of
  `_forward_swa_mqa` over a paged, padded latent cache, so every mechanism is
  reached by the layer's own gate. Gate met: `test_dots3_note_attn`,
  **30 cases / 2418 assertions**, against an independent double reference that
  takes the materialized-MHA route with no cache and a direct positional window.
  §4.7 carries the evidence, the mutation table and the two fixture defects a
  green mutation found. **No device path changed** and none of W4a's three
  refusals is lifted.
- **W4b-2 — the sliding arm ON the decode path. DONE**
  (`row/MODEL-MM-dots3-note-W4b-2`, evidence §4.8, upstream re-derived at
  `bc2d63e650`). Both attention geometries run through
  `mla::ForwardMlaAttentionBlock`, reached from `ModelRegistry::Forward`, over a
  PADDED physical KV row narrowed on read with `Tensor::Slice(2, 0, logical)` —
  upstream's `_logical_cache`, and ZERO `vt` cache ops changed.
  `vt::MlaDecodeAttention` and `vt::MlaPrefillAttention` each grew an optional
  `AttentionWindow`, whose absent state is a not-taken branch proven
  bit-identical on both ops. Two of W4a's three refusals are LIFTED (the sliding
  layer, the padded row), the `index_topk` one is KEPT and NARROWED to configs
  that have a full layer, and the per-step cache-row check is KEPT against the
  PHYSICAL row. The seam's byte-identity was re-measured on six arms in a
  separate `git archive` tree and arms 0-1 reproduce W4a's fingerprints exactly.
  **The CUDA half is COMPILED and EXECUTED**, on `orin:gpu0` (Jetson AGX Orin,
  sm_87) through an `rc` lease on 2026-08-26: both TUs compiled on two
  toolchains, and both windowed parity cases ran on the device, 0 to 49,158
  assertions for the decode case and 0 to 467,010 for the prefill one (§4.8).
  Execution is proven on sm_87 only, the ten-arch compile is compile-only, and
  this is kernel-level parity on two ops rather than the end-to-end gate. A
  windowed prefill with chunked CONTEXT is still refused by name and stays
  `## Owed` against W4b-3.
- **W4b-3 — the DSA lightning indexer's SELECTION on the device path, and the
  two debts W4b-2 named.** The split line is that the indexer shares nothing
  with the sliding window: the sliding layers carry no indexer at all
  (`self.indexer = None` / `is_sparse = False`, model.py:432-434), so lifting
  `seq_len > index_topk` is about the FULL layers and needs the indexer weights
  on device, its logits, its top-k and a SPARSE MLA attention kernel on both
  backends — none of which the window touches. It also carries the windowed
  prefill with chunked CONTEXT and the per-layer `SlidingWindowMLASpec`
  emission, whose spec TYPE landed on `main` at
  [#1960](https://github.com/mudler/vllm.cpp/issues/1960) while W4b-2 was in
  review. That emission is BLOCKED ON `KV-DSV4-MULTICACHE` W3/W4
  ([#1925](https://github.com/mudler/vllm.cpp/issues/1925)), which is the row
  that teaches the runner to carry more than one attention group; this row
  depends on it and must not duplicate it. W4b-2's
  CUDA half is NO LONGER on this list: an `rc` lease on `orin:gpu0` compiled and
  executed it on 2026-08-26 (§4.8), and `thor:gpu0` could not have gated its
  prefill half at any point, because sm_110 is outside the `fa2` feature row.
  **W4b-3c — LANDED, evidence §4.9.** Both of its units went in together,
  because the second is what makes the first reachable at its own merge commit:
  the two `vt` primitives (an OPTIONAL selected-slot arm on
  `vt::MlaDecodeAttention`, and the `vt::DsaIndexerLogits` +
  `vt::DsaTopkSelect` family) and the seam-and-model half (the indexer group on
  `MlaBlockDims` / `MlaBlockWeights`, the indexer call inside
  `mla::ForwardMlaAttentionBlock`, the per-token sparse-MQA routing on
  `MlaBlockMetadata`, the five indexer tensors in `MaterializeDots3NoteDevice`,
  and the NARROWED refusal). So `seq_len > index_topk` no longer refuses on its
  own: a step past it whose requests are all single-shot prefills is SERVED, and
  the refusal that remains is the exact complement of the route rather than a
  second predicate beside it. **What W4b-3 still owes** is the windowed prefill
  with chunked CONTEXT, the per-layer `SlidingWindowMLASpec` emission, and the
  PER-REQUEST routing of a mixed step that the narrowed refusal stands in for.
  All three are in `## Owed`.
- **W5 — MoE. LANDED, evidence §4.10.** Ungrouped `noaux_tc` at 256/8 + the one
  shared expert at `moe_intermediate_size * n_shared_experts`, through
  `Dots3NoteMoeBlock` over `vllm::RunMoePlaced`, with NO `vt` op changed. It
  carried two things beyond the block. **W5a** made materialization pick the
  MoE arm per layer — it loaded `mlp.{gate_up,down}_proj` unconditionally
  before, and on a MoE layer those tensors do not exist — and added a named
  BLOCKWISE-FP8 refusal keyed on `quantization_config.weight_block_size`, which
  is what the `-fp8` sibling carries. **W5c**
  ([#2176](https://github.com/mudler/vllm.cpp/issues/2176)) removed the nextn
  refusal, which was STRICTER THAN UPSTREAM, and replaced it with a named W10
  deferral and its own accounting bucket. Together they make
  `Dots3NoteDeviceRefusal(released_params)` EMPTY. What that does NOT mean is
  recorded in §4.10 and in the row header: the MoE is 94.62% of a 576.89 GB
  checkpoint and nothing here can hold it.
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

- **PER-REQUEST sparse routing for a MIXED step, and the refusal that stands in
  for it.** The W4b-3c review found the route predicate and the refusal
  predicate to be different predicates with a reachable gap between them, and
  the repair widened the refusal to the exact complement of
  `Dots3NoteSparseEligibility::Active`: a step in which ANY request resumes and
  ANY request exceeds `index_topk` is now refused BY NAME. **That is the
  conservative half.** Routing per request — the fresh requests served sparsely
  while a resumed request at or under `index_topk` rides the dense path it is
  entitled to, via an identity selection, which the op gate already proves is
  bit-for-bit the unselected call — is expressible, but it needs a per-request
  flag on `MlaBlockMetadata` and an identity fill inside
  `mla::ForwardMlaAttentionBlock`, and it rescues only ONE of the two halves the
  refusal covers.
  **BOTH halves are ordinary, and the record has to say so.** At the released
  geometry `index_topk` is 2048, so a co-scheduled decode whose context is at or
  UNDER 2048 tokens — the RESCUABLE half — is at least as common as one past it.
  What the refusal turns away today is therefore a genuinely common serving
  shape rather than a corner: ANY step that mixes a resumed request with a fresh
  prompt past `index_topk` is refused, on whichever side of 2048 the resumed
  request sits. **The conclusion is unchanged.** Serving that step with no
  selection, on a model whose selection prunes, is a wrong answer in silence,
  and a refusal that names the missing part is better than that. Where the two
  halves differ is who unblocks them. The rescuable half — every resumed request
  at or under `index_topk` — is THIS row's work and needs nothing from anyone
  else: a per-request flag and an identity fill. A resumed request PAST
  `index_topk` needs the indexer's own 128-wide key cache and is the half
  BLOCKED ON `KV-DSV4-MULTICACHE`
  ([#1925](https://github.com/mudler/vllm.cpp/issues/1925)). Owner: this row, a
  later W4b-3 brick. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **Four sibling `LoadF32At` copies still form a typed pointer to a tensor
  payload.** `src/vt/cpu/cpu_conv1d_depthwise.cpp`, `cpu_conv2d.cpp`,
  `cpu_conv3d.cpp` and `cpu_attn_relpos.cpp` each carry the byte-for-byte copy of
  the function this brick repaired in `cpu_layernorm.cpp`, and each is undefined
  behaviour the moment a weight or bias reaches it through `BorrowStTensorBytes`
  rather than a copy. None is reached that way today, which is why none is
  repaired here: an unreached kernel changed without a red-before is the shape
  this protocol refuses. The class already has an owner and an open issue —
  [#627](https://github.com/mudler/vllm.cpp/issues/627), "Unaligned safetensors
  reads need a checker", which asks for exactly this grep — and this brick's
  finding is its fourth recurrence. Owner: #627.
- **The two `vt` MLA-decode arms disagree on an OUT-OF-RANGE selected
  position.** `src/vt/cpu/cpu_mla_attn.cpp` refuses by name when a selected
  token position is `>= seq_len`; `src/vt/cuda/cuda_mla_attn.cu` scores the slot
  `-inf` and returns a number. On `-1` the two agree, and both files' headers
  claim the arms are behaviourally identical. It is NOT reachable from the
  current wiring — `vt::DsaTopkSelect` bounds every emitted position by
  `win_end` — and neither behaviour is gated on either arm, so which one is
  correct is a decision rather than a transcription. Both sites now carry the
  divergence in a comment. Owner: this row. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **The indexer's per-step scratch is allocated and freed inside the layer.**
  `src/vllm/model_executor/layers/attention/mla_attention.cpp` allocates
  `topk_idx [T, index_topk]` i32, `topk_cnt [T]` i32 and four more buffers per
  FULL-attention layer per step, and frees them at the end of the block. At the
  released geometry (`index_topk` 2048) an 8192-token prefill chunk is ~64 MiB
  allocated and freed thirteen times per step. Upstream instead keeps
  `topk_indices_buffer` as a PERSISTENT workspace and narrows it per step
  (`sparse_attn_indexer.py:431-432` pre-fills it, `attention.py:759` narrows
  it), which is also why its `valid_counts` matters at all. Nothing here is
  wrong; it is a per-step allocator cost on the model path, recorded so it is
  not discovered as a surprise when the first throughput axis is measured.
  Owner: this row. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **The end-to-end parity gate against vLLM** — token-exact or the ratified
  near-tie form, chosen by measurement. Blocked on §6.2 (no host we own runs the
  oracle at any published precision) and on the beyond-pin position of §6.1.
  Owner: this row. Issue [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **Every throughput, latency and memory axis.** Open by construction while the
  gate above is owed; see §6.4 for why no number is claimable meanwhile.
- **CLOSED at W4a: the full-attention layer is on the decode path, and the seam
  carries the deltas.** Both halves of W3's entry are discharged — §4.6 is the
  evidence — and the entry is kept here rather than deleted so a reader who
  followed W3's `## Owed` link lands on the answer instead of a gap.
- **CLOSED at W4b-2: the SLIDING half of everything W4a did is on the decode
  path.** 33 of the 46 layers are `sliding_attention`; both attention geometries
  now run through `mla::ForwardMlaAttentionBlock` over a PADDED physical KV row,
  reached from `ModelRegistry::Forward`. §4.8 is the evidence. The entry is kept
  here rather than deleted so a reader who followed W4b-1's `## Owed` link lands
  on the answer instead of a gap. The paragraph W4b-1 wrote here — that lifting
  the refusal needed changes inside `vt::ConcatAndCacheMla`,
  `vt::MlaDecodeAttention` and the MLA prefill gather — was the FALSE constraint
  §4.7 already corrects: the cache ops are stride-driven and ZERO of them
  changed. What the window needed was a `window_size` on two of them, which is
  the additive shape this tree uses everywhere else. Owner: row
  `MODEL-MM-dots3-note-dots3-note-for-causal-lm`. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **CLOSED at W4b-2, on `orin:gpu0`: the CUDA half of the windowed decode and
  prefill is COMPILED and EXECUTED on sm_87.** `cuda_mla_attn.cu` moves
  `kv_start` in both split stages, and `cuda_flash_attn_fa2.cu`'s MLA prefill
  launcher performs the `is_local` normalization its paged sibling already
  performs. **The two halves are named separately here on purpose, because
  W4b-2's first record merged them and the #1969 review caught it**, and an `rc`
  lease closes one at a time. DECODE: `test_ops_mla_attn`'s "CUDA mla_decode:
  the sliding window matches the CPU reference", 0 assertions with no device and
  **49,158** on the device. PREFILL: `test_ops_mla_prefill`'s "CUDA MLA prefill:
  the sliding window matches the CPU reference", which did not exist until the
  #1969 repair added it, 0 assertions with no device and **467,010** on the
  device. The control in both rows is the same binary under
  `CUDA_VISIBLE_DEVICES=""`. Both TUs also compiled under CUDA 12.6 for sm_87 and
  under CUDA 13.0 across the full CI arch list, 10 and 6 per-arch cubins read
  back with `cuobjdump --list-elf`. §4.8 carries the numbers and the recipe.
  **What stays open is SCOPE, not scheduling.** Execution is proven on sm_87
  ONLY; the ten-arch result is compile-only, because CUDA 13 cannot run against
  that box's NVRM 540.4.0 driver; and this is kernel-level parity on two ops, not
  the end-to-end model gate, which is the first entry in this list.
  **The designation of `thor:gpu0` as this row's CUDA host is CORRECTED for this
  path.** Thor's sm_110 is outside `VT_CUDA_FEATURE_TABLE`'s `fa2` row, so
  `MlaPrefillAttentionCuda` throws there and thor could never have gated the
  prefill half. Owner: this row. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **A windowed PREFILL that also carries chunked CONTEXT is refused by name.**
  Upstream caps a sliding layer's gather at `min(seq_len, query_len + W - 1)` and
  runs one varlen call per request group (`attention.py:206`, `:594-654`), so
  `forward_mha`'s LSE merge has no windowed form to mirror and the seam throws
  rather than merging an unwindowed context into a windowed suffix. Reachable in
  production by a chunked prefill of a long prompt; not reachable on the
  RELEASED checkpoint, which refuses at its first MoE layer. Owner: this row,
  **W4b-3**. Issue [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **W4b-1's `dots3_note_attn.{h,cpp}` sliding functions stay HOST REFERENCE
  code.** `ForwardSlidingAttention`, `GatherSwaKv`, `ApplySwaScoreMask`,
  `BuildSlidingWindowMetadata`, `WritePaddedMlaCache` and
  `NarrowLogicalCacheRows` have no production call site and did not gain one at
  W4b-2, because the device path reaches the same key set through the paged
  block table instead of upstream's Triton gather-plus-mask (§4.8). They are the
  gate's oracle, which is the status W3's `ForwardFullAttention` has had since
  W4a. Stated rather than left to be inferred, per `## Nothing lands dead`.
  Owner: this row. Issue [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **The 33 sliding layers report a FULL-LENGTH `MLAAttentionSpec`; upstream
  gives them a `SlidingWindowMLASpec`.** `MLAAttention.get_kv_cache_spec`
  branches on the window and returns `SlidingWindowMLASpec(..., sliding_window=
  self.sliding_window)` when one is set
  (`vllm/model_executor/layers/attention/mla_attention.py:1215-1219` @
  `bc2d63e650`), and every sliding layer sets one, because
  `Dots3NoteSlidingAttention` passes `sliding_window=config.sliding_window_size`
  into `MLAAttention` (`vllm/models/dots3_note/nvidia/model.py:457`).
  `MakeDots3NoteKVCache` emits one uniform `v1::MLAAttentionSpec` for all 46.
  **No correctness consequence** — the window is applied on READ and the W4b-2
  gate proves it — but 72% of the tower then holds a full-length latent cache
  where upstream caps a windowed layer at
  `cdiv(min(sliding_window - 1 + extra_retained + in_flight, max_model_len),
  block_size) + 1` blocks
  (`SlidingWindowSpec.max_admission_blocks_per_request`,
  `vllm/v1/kv_cache_interface.py:696-722`), i.e. 513 against 524288 on the
  released config. It is the largest memory property of this architecture and
  a token gate structurally cannot see it, which is the class `porting.md`
  names. **What is missing has shrunk from three pieces to one, and the rest
  belongs to another row.** `SlidingWindowMLASpec` LANDED at KV-DSV4-MULTICACHE
  W1 ([#1960](https://github.com/mudler/vllm.cpp/issues/1960), `c1e6f3fb9`)
  while W4b-2 was in review, so this tree carries the TYPE and the earlier text
  here calling it omitted is false.
  `SlidingWindowSpec::max_admission_blocks_per_request`, the formula quoted
  just above, is present at `include/vllm/v1/kv_cache_interface.h:358-361`.
  Only `max_memory_usage_bytes` is still absent from that header (`:64`), so the
  saving is not yet expressible as a number. **The heterogeneous per-layer GROUP
  SPLIT is NOT this row's to do.** It is `KV-DSV4-MULTICACHE` W3, with W4 for
  non-uniform block sizes. The runner today selects the FIRST non-eagle
  full-attention or MLA group as its target
  (`src/vllm/v1/worker/gpu/runner.cpp:703-712`). **This paragraph described the
  PRE-`ca3dcda21` runner until the W4b-3c review, and the description is no
  longer true.** The blanket refusal landed at `6b18829bc`
  (KV-DSV4-MULTICACHE W2,
  [#1973](https://github.com/mudler/vllm.cpp/issues/1973)); W3 (`ca3dcda21`)
  then generalized it, so a leftover group now switches the runner onto the
  multi-cache path (`:784-800`) and is ALLOCATED, and the `VT_CHECK` on
  `:860-870` refuses only the four shapes that path cannot represent — a spec
  that is neither an `AttentionSpec` nor a `MambaSpec`, a SECOND recurrent
  group, an EAGLE draft group, and a group whose published layer names do not
  all resolve to distinct in-range indices. So publishing a second
  `MLAAttentionSpec` before the wiring exists no longer throws at construction;
  it allocates a cache nothing reads. dots3-note DEPENDS on that row, must not
  duplicate it, and must not publish the group early.
  §4.8 carries the derivation. Owner: this row for the per-layer emission,
  **BLOCKED ON** `KV-DSV4-MULTICACHE` W3/W4
  ([#1925](https://github.com/mudler/vllm.cpp/issues/1925)). Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **One refusal in the device forward is UNREACHABLE and therefore untested.**
  `Dots3NoteModel::ForwardDevice`'s `VT_CHECK(ld.head_size() <= physical_row)`
  cannot fire through any production entry point, and this is recorded rather
  than dressed up as a gated refusal. Both sides come from the same parsed
  config: `physical_latent_row()` IS `swa.latent_row()`
  (`dots3_note.h:192`), so on a SLIDING layer the comparison is an identity, and
  on a FULL layer `ParseDots3NoteParams` has already refused
  `physical_latent_row() < full.latent_row()` at load
  (`dots3_note.cpp:389`, gated at
  `tests/vllm/models/test_dots3_note_scaffold.cpp:720-722`). Deleting it therefore
  leaves the gate green — MEASURED, `scripts/mutation-harness.py` at this head,
  compiler exit 0, `test_dots3_note_attn` 36 cases / 3028 assertions, SURVIVED —
  and the #1969 review's finding that it is "backstopped by `Tensor::Slice`"
  understates it, because the backstop is not reached either. It is kept as the
  executable spelling of upstream's `assert physical_head_size >= self.head_size`
  (`model.py:210`); making it load-bearing means giving the forward an input the
  loader cannot produce, which is not a shape this row wants. Owner: this row.
  Issue [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **CLOSED at W4b-3c: the six-arm DeepSeek byte-identity probe IS committed**,
  in `tests/vllm/model_executor/layers/attention/test_mla_attention_block.cpp`,
  with its arm definitions, dims, seeds and scalar values in the tree. It prints
  an FNV-1a fingerprint of the RAW OUTPUT BYTES per arm and asserts run-to-run
  stability, so a printed fingerprint is a property of the TREE rather than of
  one run. Measured in ONE session with the probe injected byte-for-byte into a
  `git archive` of the base SHA `157636cf1`: all six arms identical (§4.9). The
  paragraph below is the state this closes, kept rather than deleted because it
  is what §4.6's and §4.8's tables still refer to. **The six-arm DeepSeek
  byte-identity probe is not committed, so neither §4.6's
  nor §4.8's fingerprints can be reproduced.** Both tables are valid
  base-vs-head statements within their own session and neither is reproducible
  across sessions; §4.8 records the measurement that proves the four differing
  arms are a probe difference and not a behaviour change, and the general rule
  that a fingerprint from an uncommitted hand-written probe is not a
  cross-session quantity. Not fixed in the #1969 repair: neither scratch tree
  survives, and a third hand-written probe would produce a third set of numbers
  and no more reproducibility than two. What discharges it is committing the
  probe — its arm definitions, dims, seeds and scalar values — beside the
  DeepSeek gates, once. Owner: this row. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **CLOSED at W4b-3c for the case the indexer can serve: a long SINGLE-SHOT
  prefill is now selected SPARSELY on the device path.** The two `vt`
  primitives — an optional selected-slot arm on `vt::MlaDecodeAttention` and the
  `vt::DsaIndexerLogits` / `vt::DsaTopkSelect` pair — landed on CPU and CUDA;
  the indexer runs INSIDE `mla::ForwardMlaAttentionBlock`, where `q_c` already
  is; and `BuildDots3NoteSparseStep` promotes such a step to per-token MQA
  exactly when upstream does (`use_dense_mha = prefill_max_seq_len <=
  topk_tokens`, `sparse_mla_attention.py:296-299`). §4.9 is the evidence, the
  reachability mutation is measured, and the CUDA arm executed on `dgx:gpu0`.
  **What is still owed is the RESUMED step**, and the discriminator is the
  INDEX KV CACHE rather than the sequence length: the indexer's `k` for a token
  comes from that token's hidden state, so a step that resumes has no index key
  for its context. That cache is a second attention group on the same layers,
  which is `KV-DSV4-MULTICACHE`
  ([#1925](https://github.com/mudler/vllm.cpp/issues/1925)). The forward refuses
  it BY NAME and the refusal names that row. Owner: this row for the selection,
  **BLOCKED ON** `KV-DSV4-MULTICACHE` for the index cache. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **NEW at W4b-3c: three residues the seam carries, each measured rather than
  assumed.** (a) Upstream FUSES the indexer's `wk` and `weights_proj` into one
  `MergedColumnParallelLinear` (`deepseek_v2.py:700-707`); we issue two GEMMs,
  because the `k` half goes straight to `k_norm` and `vt::LayerNorm` requires a
  contiguous input. Identical arithmetic, one extra launch per full layer per
  step; the fold needs `vt::LayerNorm` relaxed to stride-driven, which is a
  change to a shared normalizer every pre-Llama family uses. (b) The `k_norm`
  epsilon is the upstream LITERAL `1e-6` (`:708`) and NO value gate on this
  path can hold it: mutation M18 moves it three orders of magnitude and the
  residue is unchanged to six significant figures, because the change is inside
  the bf16 store's own granularity. (c) The indexer's fp8 `q_scale`
  (`:831-838`) stays absent, because both dots3-note arms are unquantized and
  it is exactly 1 there; the op carries the field and refuses a malformed one.
  Owner: this row. Issue [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **SUPERSEDED by the entry above — kept so a reader who followed W4a's or
  W4b-2's `## Owed` link lands on the answer rather than a gap.** W3 ported
  the selection maths as a host reference and W4a did not wire it: the shared
  MLA seam computes DENSE attention, which is upstream's answer only while
  `context + query <= index_topk`, because the top-k then selects every causal
  candidate. Past that bound the device forward REFUSES BY NAME rather than
  serving dense attention on a sparse model — W3 measured a wrong selection at
  0.392 on the layer output, so the gap is loud rather than latent. It is a real
  ceiling on what W4a can serve: 2048 keys against a 524288-position model.
  **W4b-1 did NOT lift it, and it could not have**: the sliding layers carry no
  indexer at all (`self.indexer = None` / `is_sparse = False`, model.py:432-434),
  so nothing W4b-1 wrote touches the FULL arm's selection, and
  `Dots3NoteSlidingAttnDimsFrom` REFUSES a params object whose sliding arm claims
  one. **W4b-2 did not lift it either, and it could not have, for the same
  reason** — but it NARROWED who is asked: the per-step bound is now checked only
  for a config that HAS a full-attention layer, so a pure-SWA schedule is no
  longer refused for a mechanism it does not carry. §4.8 records the
  measurement. **The indexer's KEY CACHE is a second cache kind on the same
  layers, so that half carries the same dependency as the sliding-window spec
  above**: the runner picks ONE target attention group
  (`src/vllm/v1/worker/gpu/runner.cpp:703-712`), and since `ca3dcda21` a second
  non-eagle `AttentionSpec` group is ALLOCATED by the generalized multi-cache
  path rather than refused — the `VT_CHECK` on `:860-870` keeps only four
  shapes, and a second MLA group is not one of them. That makes publishing the
  index cache here MORE dangerous than it was, not less: it would be allocated
  with nothing reading it. Carrying the group and its wiring is
  `KV-DSV4-MULTICACHE` W3/W4
  ([#1925](https://github.com/mudler/vllm.cpp/issues/1925)), not this row.
  The SELECTION maths, the logits kernel, the top-k and the sparse MLA kernel
  are this row's. Owner: this row, **W4b-3**, for the selection; **BLOCKED ON**
  `KV-DSV4-MULTICACHE` W3/W4 for the index cache group. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **The PADDED physical latent row.** `MakeDots3NoteKVCache` already reports the
  1088-wide row both classes share, and W4a refuses any config whose physical row
  exceeds the full layers' logical 576 — at CONFIG level, so the loader does not
  materialize a tower the forward then rejects — plus a per-step check for a
  cache the engine sized differently from its own config. Narrowing a padded row
  on read is `Dots3NotePaddedSparseImpl._logical_cache`. **W4b-1 ported its
  SEMANTICS and NOT its wiring** — `PaddedMlaCacheSpec`, `WritePaddedMlaCache`
  and `NarrowLogicalCacheRows`, gated on an exact round-trip, on the tail of
  every physical row staying untouched by a logical-width write, and on a
  logical-stride reader differing by 7.99 over the same buffer. **Both device
  refusals stand**, the config-level one and the per-step one — and they stand
  BY SCOPE CHOICE rather than by constraint, which is the correction §4.7
  records. `vt`'s MLA cache ops are STRIDE-DRIVEN, `Tensor::Slice(2, 0, logical)`
  is upstream's `kv_cache[..., : head_size]`, and a probe wrote, gathered and
  decoded through a physical-7 / logical-5 view at 30/30 with no `vt` change at
  all. **CLOSED at W4b-2**: the config-level refusal is gone, the narrowing is
  one `Tensor::Slice(2, 0, logical)` in `Dots3NoteModel::ForwardDevice`, no `vt`
  op changed, and the gate reads the RAW cache bytes after a real forward to
  assert the pad lanes of every full-layer slot are untouched (§4.8). The
  PER-STEP refusal stays and is not the same check: an engine allocates the
  cache separately from the config it was built from, so a row that disagrees is
  an input only the forward can see. Owner: this row. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).
- **The nextn tail on the device path.** W4a refuses a config with
  `num_nextn_predict_layers > 0` rather than enumerating, loading and never
  running the extra block. `Dots3NoteMTPModel` over the speculator seam is
  **W10**, which also still owes the reconciliation W1/W2 could not make:
  `config.layer_types` has no entry at the nextn index. Owner: this row. Issue
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
- **The bf16 memory format of the four deltas — HALF CLOSED at W4a, and the
  residue is a MEASURED number rather than an unknown.** W3's reference is
  double throughout, which is strictly wider than the model path, and could see
  neither width. The device path settles both: the activation stream is bf16
  end to end, both LoRA rescales are a bf16 `vt::MulScalar` exactly as upstream
  multiplies a bf16 tensor by a python float (`model.py:155`, `:159`), and the
  headwise gate's LOGIT is bf16 like upstream's (`g_proj` carries no
  `params_dtype`, `model.py:292-297`) and its sigmoid is computed in **fp32**
  (`model.py:196`).
  **The logit width was a SECOND unmirrored step until the W4a review**, which
  found an f32 GEMM output on a model path — the too-WIDE case `porting.md`
  says a token gate cannot catch. Narrowing the GEMM closed it, and §4.6 shows
  why no gate here could have: rounding the logit moves the gate by at most
  `0.2239 * 2^-9 = 4.38e-4`, under the bf16 store's own `2^-9` half-ulp, so
  mutation M16 reverting the narrowing comes back GREEN by construction.
  **What is still owed is ONE rounding step.** Upstream rounds the sigmoid to
  the activation dtype BEFORE the multiply (`torch.sigmoid(gate.float()).to(
  attn_out.dtype)`) and then multiplies in bf16, so its product is rounded
  twice; `vt::SharedExpertGate` keeps the sigmoid in f32 and rounds only the
  product. §4.6 measures the difference at **3.906e-3 over a scale of 0.9453**,
  i.e. under 2^-7 relative, on this fixture — a measurement, not a bound. The
  same convention already ships for Qwen3.6's shared-expert gate. Mirroring it
  exactly needs an op whose store dtype is the caller's. Owner: this row. Issue
  [#699](https://github.com/mudler/vllm.cpp/issues/699).

- **`vt::QuantFp8Group` has no `use_ue8m0` rounding, and it is owed against
  W9** (recorded at W5, #699). It does NOT bite at W5: it is the ACTIVATION
  quantizer and W5 is entirely on the bf16 path, so nothing in that brick calls
  it. It probably does bite at W9, because upstream's blockwise-FP8 MoE routes
  through DeepGEMM with e8m0 scales
  (`vllm/models/dots3_note/nvidia/vision_moe.py`'s own docstring names the
  sibling trap on the vision side: "the native NOTE encoder keeps dynamic
  activation scales as FP32 instead of rounding them to E8M0"). A port that
  quantizes activations with plain scaling there disagrees with the kernel
  upstream runs, and the disagreement is numerically silent. Recorded here with
  the reason so W9 does not re-derive it.
- **The blockwise-FP8 arm itself is owed to W9**, and it is now refused BY NAME
  rather than by a bare tensor miss (W5a). `dots-studio/dots3-note-prev-fp8` @
  `7c14222e22423d6df6848eb0d1c5c3a88a00311a` carries
  `quantization_config.weight_block_size = [128, 128]` and ships a
  `weight_scale_inv` beside every projection; `dense_loaders::MaterializeBf16Source`
  reads a per-tensor or per-output-ROW `<name>_scale` and nothing else.
- **The GROUPED (CUDA) arm of `Dots3NoteMoeBlock` has no execution evidence.**
  `vt::MoeGroupedGemmBf16` and `vt::MoeGroupedGemmBf16GateUpSilu` are registered
  for CUDA only, and every dots3-note model case builds a `kCPU` `vt::Queue`, so
  the CPU gate takes the reference arm and the `orin:gpu0` lease measured a ZERO
  device-versus-control delta on every W5 case. The arm is UNGATED rather than
  dead: production reaches it whenever the engine queue is CUDA. Closing it needs
  a device-queue bench — a `kCUDA` queue, a device-memory `PagedKvCache`, and the
  resident uploads that follow — which is test infrastructure no brick on this
  row has built, and which W4b-2 and W4b-3c also did not (§4.8: their CUDA
  evidence is kernel-level parity on two ops, not the model).
  **UNGATED AND NOT DEAD, proven statically rather than asserted** (W5 fresh
  review, #2187). All three of `Dots3NoteGroupedMoeEligible`'s conditions hold
  on a CUDA build: `vt::OpRegistered(kMoeGroupedGemmBf16, kCUDA)` is true
  (`src/vt/cuda/cuda_matmul_nvfp4.cu:2722`), `w.expert_gate` is non-empty on
  every MoE layer, and `t->nk` is FALSE on all three expert tensors because
  `LoadBf16Transposed` returns a `MakeOwned` tensor and never sets the flag
  (`include/vllm/model_executor/models/dense_weight_loaders.h:388-401`). So the
  arm is reached the moment the engine queue is CUDA, and the missing evidence
  is a measurement rather than a wiring question.
- **The FIX for the address-keyed residency defect is itself only structurally
  gated** ([#2193](https://github.com/mudler/vllm.cpp/issues/2193), W5 fresh
  review F1). W5 first shipped `Dots3NoteMoePtrsFor` as a process-lifetime
  `static std::map<const Dots3NoteMoeWeights*, Dots3NoteMoePtrs>`, which is the
  shape #237 removed from `qwen3_5.cpp` in `ce2349dee`, and cited that repair as
  its warrant. It is repaired here — `Dots3NoteMoeWeights` owns a
  `ResidentSlot resident_moe` and the accessor builds into it under a mutex,
  the `laguna.cpp:497-507` shape — and
  `tests/vllm/models/test_moe_resident_lifetime.cpp` gained four cases for this
  block, including the placement-new address-reuse case. **What those cases
  CANNOT see, said rather than implied.** The accessor is file-local to
  `dots3_note_device.cpp` and is called only from inside the grouped arm above,
  which is CUDA-only with no CPU reference tier, so no CPU gate can call it.
  The cases pin the invariant the fix rests on (residency is a member of the
  weights, so it is per-object and cannot be inherited through a reused
  address); a mutation that reverts the accessor BODY while leaving the member
  in place is not observable from the CPU and rides the same device run this
  arm already owes. `deepseek_v2.cpp`'s `MoePtrs` (`04f5c01e7`, 2026-07-22)
  still carries the pre-#237 shape; it is a SACRED path, is deliberately NOT
  touched here, and is owed under #2193 until a row picks it up.
- **The quantization refusal keys on `weight_block_size` alone**
  ([#2190](https://github.com/mudler/vllm.cpp/issues/2190), W5 fresh review F5).
  `quant_method` is parsed (`dots3_note.cpp:264-268`) and stored
  (`dots3_note.h:206`) and is read for nothing but the text of the blockwise
  message, so a config with `quantization_config.quant_method = "fp8"` (or
  gptq/awq/mxfp4) and NO `weight_block_size` passes
  `Dots3NoteDeviceRefusal` and `MaterializeBf16Source` silently dequantizes a
  per-tensor or per-row `_scale` into a bf16 GEMM. That is the case the
  refusal's own comment names as the worse one, five lines above the branch that
  does not cover it. No released checkpoint is affected. Owner: this row, W9.
- **`hidden_act` is not mirrored**
  ([#2191](https://github.com/mudler/vllm.cpp/issues/2191), W5 fresh review F6).
  `DeepseekV2MoE.__init__` raises `ValueError("Unsupported activation ... Only
  silu is supported for now.")` at `deepseek_v2.py:310-314` @ `bc2d63e650`,
  inside the very `__init__` this brick ports, and `grep -c hidden_act` over
  `dots3_note.{cpp,h}` is 0 — so a non-silu config runs SwiGLU silently. The
  same hole is in `deepseek_v2.cpp`, so it is a mirror gap rather than a W5
  regression, and the released config is silu. Owner: this row.
- **The W1/W2 accounting fixture can no longer be MATERIALIZED**, and that is a
  consequence of W5 rather than a defect. Those gates drive all 38006 names
  through the production loader from a synthetic checkpoint of ONE-ELEMENT
  tensors, which worked only because the released config was refused and
  materialization was skipped. It is not skipped now, so the loader accounts for
  every name and then refuses the first WEIGHT SHAPE. The cases assert that
  discrimination instead, which is the same statement one step later — an
  unaccounted, missing or duplicated name throws a DIFFERENT message strictly
  earlier. TWO of the three are separately gated — UNCLAIMED and MISSING each
  have a subcase in "the unported arms REFUSE BY NAME"; DUPLICATED has none, and
  no fixture can give it one, because `acc.duplicated` is filled at
  `dots3_note.cpp:624` when `EnumerateDots3NoteTensors` emits the same name
  twice, which is a property of the ENUMERATOR and not of the checkpoint
  (W5 fresh review F7). A shape-true fixture for this
  config starts at a 1.5 GiB `embed_tokens` and is not buildable in a test.

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

**W3's own closing read "the maths is not reachable", and W4a is why that
sentence is now historical rather than current.** It is kept above, unedited,
because `main` is never rewritten and because the reason the row stays `SPIKE`
has moved once more — see below.

**W4a — DONE.** The full-attention layer is ON THE DECODE PATH.
`mla::ForwardMlaAttentionBlock` — the block DeepSeek-V2 decodes through under a
SACRED token-exact gate — grew the four optional fields the three non-indexer
deltas and the headwise gate need, and `Dots3NoteModel::ForwardDevice` became a
real forward for one config shape: every layer `full_attention` with a dense
MLP, reached through `ModelRegistry::Forward` over the real loader and a real
synthetic checkpoint. **The DeepSeek path is byte-identical before and after**,
measured over the raw output bytes of SIX fixed batches spanning the seam's whole
branch space — q_lora present and absent, both rope layouts, both dtypes — with
the base arm built in its own `git archive` tree at `d7d1ee914`, not argued from
the defaults. §4.6 carries that table, the 18-row mutation table and the W4a/W4b
split.

**Three things W4a measured rather than assumed, and the last two are cautionary.**
The headwise gate's widths are now answered: the LOGIT is bf16 like upstream, and
the one remaining unmirrored rounding is bounded at 2^-7 relative on the gated
output. Mutation M5 first read GREEN — the k_pe-norm ORDER defect moved the
measurement to 0.0193 against a 2e-2 bound and slipped underneath — and the
FIXTURE was sharpened until the defect is visible rather than the bound widened.
And the fresh review found the SAME disease a second time in the same file: at
those ranks a mutation dropping `q_lora_scale` alone reddened only 4.8% over the
bound. The fixture's LoRA ranks now match the released model's ratio, that
mutation sits **15.2x above the bound** (and 42.6x above the residue, which is a
different pair and is quoted as one), and all three ratios are tabulated in §4.6
so none of them can be read as another's margin.

**W4b-1 — DONE, and W4b is SPLIT.** The sliding arm's maths exists. The
geometry (64 heads, kv_lora 1024, qk_nope 192, theta 5e4, window 513, no
indexer) resolves from the released `config.json`; the whole §2.3 machinery is
ported as host code — `SwaGatherLen`, `GatherSwaKv`, `ApplySwaScoreMask`,
`BuildSlidingWindowMetadata`, and the `PaddedMlaCacheSpec` / `_logical_cache`
pair; and `ForwardSlidingAttention` computes the layer the way upstream does,
the ABSORBED MQA of `_forward_swa_mqa` over a paged, padded latent cache, so
every mechanism is reached by the layer's own comparison rather than only by its
unit case. **Gate met: `test_dots3_note_attn`, 30 cases / 2418 assertions**,
against an independent double reference that takes the other route at four
levels — materialized MHA, no cache, a direct positional window predicate, a
max-subtraction-free `long double` softmax — agreeing to 1.2e-16 — 5.0e-16.
RED-first: with every mechanism neutralised at once the same gate reads 5 cases
/ 68 assertions failed. §4.7 carries the evidence and the 26-row mutation table.

**The split line, stated plainly because the row's credibility rests on scope
statements being exact.** W4b-1 is SEMANTICS; W4b-2 is the DECODE PATH. **No
device path changed here and none of W4a's three refusals is lifted** — a
sliding layer, a MoE layer, a padded physical row and a nextn tail are still
refused at config level, and a request past `index_topk` and a disagreeing cache
row are still refused per step. The gate's last case asserts THREE refusals
executably — MoE, `sliding_attention` and the padded row — of which one,
the padded row, is among W4a's three; the other two of W4a's three stay asserted
by W4a's own unchanged case, which this brick did not touch.

**The reason the line falls where it does, corrected after the fresh review
refuted the first one by execution.** The padded row is deferred **by scope
choice, not by constraint**: `vt`'s MLA cache ops are STRIDE-DRIVEN,
`Tensor::Slice(2, 0, logical)` IS upstream's `kv_cache[..., : self.head_size]`,
the tree already gates a strided cache view at
`tests/vt/test_ops_mla_cache.cpp:259`, and a probe wrote, gathered and decoded
through a physical-7 / logical-5 view at 30/30 with ZERO `vt` changes. The real
constraint is the WINDOW: `vt::MlaDecodeAttention` attends the whole `seq_len`
(`cpu_mla_attn.cpp:94`) with no window and no per-slot `valid`, and neither
argument struct carries a window field — so a windowed decode and prefill is a
new kernel on both backends and owes the seam byte-identity W4a produced.

**Three mutations came back GREEN and the FIXTURE changed each time, never the
bound.** `gather_start` was unreachable in a prefill-shaped bench; the sliding
`qk_head_dim` equalled its `latent_row`; and the two head counts were equal. All
three are now separated and PINNED by the geometry case. That is spec §4.6's
review finding F1 applied before a reviewer had to find it — and one mutation
FAILED TO BUILD under `-Werror=unused-variable`, was recorded as `NOT A RESULT`
rather than as a pass, and re-run.

**Next dispatchable: W4b-2 — the sliding arm ON the decode path.** A `vt` MLA
cache whose physical row is wider than the row a layer reads, a windowed decode
and prefill through the shared MLA seam, and the three refusals above. All are
in `## Owed`.

**W5 — LANDED, and the row's headline changed.** The 45 MoE layers are on the
decode path through `Dots3NoteMoeBlock` over the shared `vllm::RunMoePlaced`
seam, and `Dots3NoteDeviceRefusal` returns `""` for the RELEASED
`dots-studio/dots3-note-prev` `config.json` for the first time on this row. Two
branches went: the MoE layer (W5) and the nextn tail (W5c,
[#2176](https://github.com/mudler/vllm.cpp/issues/2176)), the second of which
was a DEFECT rather than a gap — it was stricter than upstream, which drops
those weights from the main model instead of refusing. Evidence is §4.10.

**Say the other half in the same breath.** Representable is not runnable. The
MoE alone is 545.82 GB of a 576.89 GB checkpoint (94.62%), measured over the
committed headers-only index; the routed experts are 543.58 GB of that. No host
this project reaches holds it, the 298.67 GB fp8 sibling does not fit either and
is refused BY NAME as W9, and no tensor byte of either has ever been
downloaded. The gate is a consistency gate against an independent
double-precision reference, not a correctness claim against vLLM (§6.4 option
B). `supports_multimodal` went TRUE -> FALSE in the same change, because the
released config becoming loadable made a claim this port cannot honour: the
2195 vision and 430 audio tensors are named W6/W7 deferrals and the multimodal
front end (W8) does not exist. W8 flips it back.

**Next dispatchable: W6 — the vision tower**, or W9 for the quantized arms if
the fp8 sibling is wanted before the towers. `## Owed` is unchanged except that
`vt::QuantFp8Group`'s missing `use_ue8m0` rounding is now recorded against W9
with the reason, because upstream's blockwise-fp8 MoE routes through DeepGEMM
with e8m0 scales.
