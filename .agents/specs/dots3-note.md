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

**RE-MEASURED at `9035151d6` by W7a.** Every anchor below was read again in
`~/_git/vllm` at that SHA and eight of them had drifted. The drift is uniform
`+9` and it starts at `DotsSpeechEncoder`: everything above that class was
right, everything from it down was nine lines low, and the file length was nine
lines long. The corrected values are the ones written here; the previous ones
are kept in this sentence so a reader who finds them in an older commit can see
that they were repaired rather than invented.

`nvidia/audio_encoder.py` (**736**, was 745) — `RotaryEmbedding:47`,
`WhisperPositionalEmbedding:184`, `WhisperAttention:196`,
`WhisperEncoderLayer:310`, `DotsSpeechEncoder:`**`428`** (was 437) with three
stems (`_forward_conv2d_stem:`**`564`** was 573, `_forward_conv1d_stem:`**`585`**
was 594, `_forward_latent_stem:`**`598`** was 607) and
`_temporal_mask:`**`529`** (was 538). `nvidia/audio.py` (305) is the vLLM-side
wrapper.

**`DotsSpeechEncoder` builds NO positional embedding under `use_rope`, at
`audio_encoder.py:498-510`.** #2703 and §"What the two towers actually ship"
below both cite `:507-519` for that, which is the same `+9` drift applied to a
range; `498-510` is the measured one and `:508` is the
`self.embed_positions = None` line itself.

### 2.6 Multimodal front end and MTP

`nvidia/multimodal.py:`**`49`** `Dots3NoteForCausalLM(nn.Module,
SupportsMultiModal, SupportsPP)` — `get_placeholder_str:`**`65`**,
`_process_image_input:`**`144`**, `_process_audio_input:`**`156`**,
`_process_video_input:`**`172`**, `embed_multimodal:`**`225`**,
`get_mm_mapping:`**`300`**. `common/processor.py` (811) and `common/video.py`
(497) carry the prompt-side expansion and frame sampling.

**RE-MEASURED at `9035151d6` by W7a, same as §2.5, and this paragraph's drift
was NOT uniform** — it ran 16, 16, 29, 30, 30, 30 and 25 lines, which is the
signature of anchors read in a different revision rather than of one insertion
above them. The previous values were `:65`, `get_placeholder_str:81`,
`_process_image_input:173`, `_process_audio_input:186`,
`_process_video_input:202`, `embed_multimodal:255`, `get_mm_mapping:325`. The
file is 304 lines at this SHA.
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
7. **GGUF k-quant arm.** Per AGENTS.md the arm is owed, not optional; an
   unimplemented arm refuses with a message naming the missing piece.
   **CORRECTED IN PLACE, 2026-09-04 (W9a, §4.19.2).** This item used to say
   llama.cpp has no `dots3_note` architecture, so the converter is ours to write
   and there is no quant-matched llama.cpp comparison for this row. All three
   claims are false: llama.cpp merged `LLM_ARCH_DOTS3NOTE -> "dots3note"` on
   2026-08-21, `conversion/dots3.py` registers `Dots3NoteForCausalLM` by name,
   and published artifacts exist. What is owed is our LOADER arm and a pin
   advance off `b10451` (W9b/W9c/W9e/W9f), not a converter written from
   nothing.

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
`nvidia/audio_encoder.py:498-510` (W7a re-measured; this said `:507-519`, the
same `+9` drift §2.5 records) `DotsSpeechEncoder` sets
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

### 4.11 W6a puts the DENSE vision tower on a SERVED request, and refuses the pyramid by name

W6a is the first brick on this row whose output a client can ask for. Every
brick before it ended at a `ctest` binary: W3 wrote host maths, W4a/W4b put the
two attention geometries on the decode path, W5 put the MoE layers there. None
of them could be reached from `ApiServer::handle_chat_completions`, because
until #2398 and #2481 landed there was no engine path by which a vision tower
could be fed from a production entry point at all. Both are on `main` now, so
the remaining work was this model's own half: a tower, a processor, a chat
registration and the two `ModelFactory` hooks.

**The oracle, and the SHA every anchor here was read at.** vLLM is the oracle
and no secondary oracle is admissible, because vLLM implements this tower. It is
BEYOND our parity pin: `5559679229bc961848b121ccdeaa8fa5d79bec98` has no
`dots3_note` directory at all. The sources were read in the local clone
`~/_git/vllm` at **`9035151d6`**, the merge of
[vllm#51255](https://github.com/vllm-project/vllm/pull/51255) that added them:

| Upstream | Lines at `9035151d6` | What W6a ported from it |
|---|---|---|
| `vllm/models/dots3_note/nvidia/vision.py` | 677 | `DotsMoEVitConfig:27`, `RMSNorm:107`, `DotsSwiGLUFFN:126`, `DotsPatchEmbed:302`, `MoEVisionBlock:334`, `PatchMergerAdapter:441`, `DotsMoEVitModel:492` (`get_pos_ids_by_grid:566`, `rot_pos_emb:604`, `_build_single_temporal_cu_seqlens_from_grid:625`, `forward:634`) |
| `vllm/models/dots3_note/nvidia/vision_attention.py` | 477 | `rotate_half:33`, `apply_rotary_pos_emb_vision:39`, `VisionRotaryEmbedding:52`, `_RMSNorm:97`, `_VisionAttentionBase:134` (`_qkv_with_rope:149`), `VisionAttentionV2:207`, `apply_vision_attention_residual:436` |
| `vllm/models/dots3_note/common/processor.py` | 811 | `IMAGE_START/PAD/END:41-43`, `Dots3NoteImageProcessor:63` (`resized_size:97`, `preprocess:147`) |
| `vllm/models/dots3_note/nvidia/multimodal.py` | 304 | `hf_to_vllm_mapper:54-62` (`vision_encoder.` -> `visual.`), `get_placeholder_str:65`, `_process_image_input:144` |
| `vllm/models/dots3_note/nvidia/vision_moe.py` | 149 | NOTHING. It is the W6b/W9 arm and is refused by name. |

**Every anchor above names the SHA it was read at, because upstream has already
moved.** `vision_attention.py` is 477 lines at `9035151d6` and 494 lines at vLLM
`main` `7a100bb61`. An anchor read in the wrong tree is a recorded failure mode
on this project, and a `file:line` with no revision beside it is one.

#### 4.11.1 The geometry, and one word in #2512 that the fixture corrects

Every number below was read from the COMMITTED fixture
(`tests/vllm/models/fixtures/dots3_note_prev/config.json` and
`index_full.json`), not from the issue text:

| `vision_config` key | Released value | Consequence |
|---|---|---|
| `embed_dim` | 1536 | tower width; `head_dim` = 1536/24 = 64 |
| `num_attention_heads` | 24 | |
| `num_hidden_layers` | 42 | 25 dense + 17 MoE |
| `intermediate_size` | 4224 | the DENSE SwiGLU width |
| `moe_intermediate_size` | 2112 | W6b's, unread here |
| `patch_size` | 14 | patch row is 3*1*14*14 = 588 wide |
| `temporal_patch_size` | 1 | `DotsPatchEmbed.forward` takes `[:, :, 0]` of a one-deep temporal axis |
| `spatial_merge_size` | 2 | |
| `rms_norm_eps` | 1e-05 | |
| `use_bias` | false | `attn.qkv`, `attn.proj` and every `mlp.fc*` carry NO bias |
| `use_qk_norm` | true | `q_norm`/`k_norm` [64], per head, BEFORE rope |
| `is_causal` | false | the attention is bidirectional |
| `post_norm` | true | `post_trunk_norm` exists |
| `pre_pixel_shuffle` | true | the PREPROCESSOR emits 2x2-grouped patch rows and RoPE regroups to match |
| `adapter_type` | `"patch_merger"` | `PatchMergerAdapter`, NOT `PixelShuffleAdapter` |
| `adapter_in_dim` / `adapter_out_dim` | 1536 / 5120 | 4x1536 = 6144 folded to the text tower's 5120 |
| `pyramid_num_routed` | `[-1 x 25, 4, 8, ..., 60, 64, 64]` | `is_moe` is `pyramid_num_routed[i] > 0` (vision.py:363-366), so -1 is DENSE |

**One word in #2512 needs correcting, and the fixture is what corrects it.** The
issue's scope prose says the dense arm is "`post_trunk_norm` -> pixel shuffle ->
`adapter`". The released `vision_config` sets `adapter_type: "patch_merger"`,
and `PatchMergerAdapter` is upstream's own name for the arm that **skips the
pixel-shuffle permutation** and instead views every 4 consecutive 2x2-grouped
tokens as one row (`vision.py:465-471`, its docstring). The 2x2 regrouping has
not disappeared; `pre_pixel_shuffle: true` moved it into the PREPROCESSOR
(`processor.py:185-197`, the nine-way reshape and the `(0,3,6,4,7,2,1,5,8)`
transpose) and into the RoPE position builder
(`get_pos_ids_by_grid:565-574`, `rope_merge_size = spatial_merge_size`).

This is not a disagreement about geometry, and it did not need escalating.
#2512's own tensor inventory says `adapter.{ln_q, mlp.0, mlp.2}`, which is
`PatchMergerAdapter`'s state dict and nothing else — `PixelShuffleAdapter`
spells its parameters `proj.0` / `proj.1` / `proj.3` (`vision.py:423`, `:432-437`). The
inventory is right and the prose word is loose. W6a implements
`patch_merger`, and `pixel_shuffle_mlp` is REFUSED BY NAME rather than
silently mapped onto it, because the two produce different token orders from
the same pixels and neither shape-checks against the other.

#### 4.11.2 The 235 dense tensors, counted

Of the 2195 `vision_encoder.*` tensors in the released index, W6a's arm claims
**235** and refuses **1960**:

```
dense blocks 0..24, 9 each     225   norm_1, norm_2, attn.{qkv,proj,q_norm,k_norm},
                                     mlp.{fc1,fc2,fc3}          (all BF16)
patch_embed                      3   proj.weight [1536,3,14,14], proj.bias [1536],
                                     norm.weight [1536]
post_trunk_norm                  1
adapter                          6   ln_q.{weight,bias} [1536],
                                     mlp.0.{weight,bias} [6144,6144]/[6144],
                                     mlp.2.{weight,bias} [5120,6144]/[5120]
                              ----
                               235
MoE blocks 25..41 (W6b)       1960   17 x {norm_1, norm_2, attn x4, gate_weight,
                                     router_bias} = 136, plus 608 experts x 3 = 1824
                              ----
                              2195
```

The 608 is the sum of `pyramid_num_routed[25..41]`, and it is what makes the
released checkpoint still refuse.

#### 4.11.3 What W6a refuses, BY NAME

The released `dots-studio/dots3-note-prev` has 17 MoE ViT blocks, so a load of
it REFUSES at the vision tower and names W6b. **That is correct and it is this
row's established pattern rather than a new exception.** W3 refused the language
tower's MoE layers by name for four bricks before W5 lifted it; the vision
tower is at W3's stage, not at W5's. Refusing is what stops the port from
serving a tower whose pyramid it silently skipped, on a row that §6.4 records as
having no oracle to catch it.

| Refused | Named brick | Where |
|---|---|---|
| any block with `pyramid_num_routed[i] > 0` | **W6b** | `Dots3NoteVisionRefusal` |
| `quantization_config.weight_block_size` on the vision tower | **W9** | `Dots3NoteVisionRefusal` |
| `audio` modality | **W7** — see the correction below | `EncodeMmDots3Note`, and the chat seam's `allowed_limits`. **LIFTED by W7a (§4.14)** for a 16 kHz mono PCM16 WAV at or under `chunk_seconds`, and by W7b (§4.15) for one of ANY length; what remains refused is named in §4.14.5 as W7a left it, with the `chunk_seconds` row struck through |
| `video` modality | **W8** — see the correction below | `EncodeMmDots3Note`, and the chat seam's `allowed_limits` |
| `adapter_type == "pixel_shuffle_mlp"` | W6b | `ParseDots3NoteVisionParams` |
| `post_norm == false` | W6b | `ParseDots3NoteVisionParams` |
| `use_bias == true` | W6b | `ParseDots3NoteVisionParams` |
| `temporal_patch_size != 1` | **W8** (video) — this row read `W7 (video)` until W8a, two lines above the correction paragraph that is about exactly this | `ParseDots3NoteVisionParams` |
| `adapter_out_dim != config.hidden_size` | none — unservable | `Dots3NoteVisionRefusal` |
| `adapter_merge_size != spatial_merge_size` | none — unservable | `Dots3NoteVisionRefusal` |

**W7 IS AUDIO AND W8 IS VIDEO, and this table had them the other way round.**
The loader has said so since W2 — `Dots3NoteDeferredTowers()` registers
`audio_encoder.` against brick `"W7"` and describes it as "the `dots`
Whisper-variant audio tower (nvidia/audio_encoder.py)" — and #2703 is titled
"dots3-note W7: the audio tower". Two PRODUCTION refusal messages disagreed with
both: `dots3_note_registry.cpp`'s encoder refusal read "VIDEO ... is W7; AUDIO
... is W8", and `mm_chat_dots3note.cpp`'s limit comment read "Video is W7's and
audio is W8's". W7a repairs all four surfaces — this table, the two messages and
that comment — to the loader's polarity. Nothing about what is implemented
changed; a user reading the refusal was being sent to the wrong brick.

**The last two rows name no brick, and that is the point of them.** They are not
capabilities owed to a later brick; they are configs no dots3-note tower can be
served under at all. They are here because the fresh review of #2523 measured
the refusal predicate to be a strict SUBSET of the `VT_CHECK`s
`EncodeMmDots3NoteForCausalLM` makes on a served request. Three of those checks
— the adapter width against the text width
(`dots3_note_registry.cpp`), the emitted row count against the placeholder span,
and `L % merge_unit` inside the tower — were reachable from an all-dense config
the seam ACCEPTED. Reaching any of them throws inside the engine's busy loop,
which sets `AsyncLLM::errored_` permanently (`async_llm.cpp:584-601`), so the
server starts, text works, the first image request 500s, and every later
request is dead for the life of the process. That is the exact cascade the
factory-side refusal was introduced to remove, still reachable through a
narrower door. **A refusal and its route predicate must be the SAME predicate**,
which is this row's second recurrence of the finding: the W4b-3c review made it
about sparse routing, recorded in the first `## Owed` entry.

A tautology went with them. `adapter_merge_size**2 * adapter_in_dim !=
merged_dim()` read as a cross-key check and was `x != x` — `merged_dim()` is
that product, reordered (`dots3_note_vision.h`) — so it could never fire, and it
was the only refusal in the table that named no brick because it stood for no
condition. The `adapter_merge_size` row above is the real cross-key check that
belongs in its place.

A refused tower leaves the 2195 `vision_encoder.*` tensors in the accounting's
existing `vision` bucket as a NAMED deferral, exactly as before, so every W2
count assertion is byte-for-byte unchanged. What moved is the deferral's
`brick` field: `W6` -> `W6b`, because W6a is landed and W6b is what is owed.

#### 4.11.4 The gate is a CONSISTENCY gate, and says so

§6.4's option B stands. The checkpoint is 298.67 GB fp8 / 576.89 GB bf16 against
119-122 GiB hosts, so vLLM cannot be run on it here and no denominator exists.
Correctness for the tower is therefore argued by an **independent in-test
double-precision reference** written from `vision.py` / `vision_attention.py` at
`9035151d6`, sharing NO helper with the implementation, with RED-first mutation
proof.

**That establishes two implementations agree. It does not establish that either
matches vLLM.** No performance number is claimable on any axis while B holds,
and none is claimed. The reference and the implementation differ deliberately at
every step that has a choice: the reference is a scalar `double` loop with its
own softmax, its own rope and its own norms; the implementation is
`vt::MatmulBT` / `vt::RmsNorm` / `vt::RopeFromCache` / `vt::AttentionDenseFlash`
over bf16 device buffers through the shared seams.

**One formula difference is deliberate and is recorded rather than hidden.**
Upstream's vision `RMSNorm.forward` (`vision.py:114-116`) casts the normalized
value back to the activation dtype BEFORE multiplying by the weight; `vt::RmsNorm`
keeps f32 through the weight multiply and rounds once on the store (its own
header says so). Using the shared op is the seam rule. The reference does NOT
copy the cast: at infinite precision the two are the same function, so a double
reference is the algebra BOTH implement, and the gate's tolerance covers our
bf16 storage and upstream's intermediate cast together. Copying the cast into
the reference would make the reference agree with a rounding choice instead of
with the maths. §4.11.6 records the measured deviation, the bound, and the
mutation that proves the bound is not a mute switch.

**The discrete-selection rule does not bind W6a and that is a fact about the
arm, not an omission.** The dense blocks have no top-k anywhere: routing is
W6b's. When W6b lands it owes a SET-equality assertion on the router's top-k
plus the printed minimum decision margin, because a tolerance alone cannot see a
bimodal selection flip.

**The memory format is asserted against upstream explicitly** (porting.md), and
this row has already been bitten on that axis: W2's F1 fixture row proves a
re-typed `router_bias` fires. Every dense vision tensor is BF16 on disk and BF16
in the resident tower, and the gate asserts the tower's stored dtype rather than
only its values — a token gate cannot see a dtype that is too wide.

#### 4.11.5 Reachability — the production entry point, and what it costs to fake

The production entry point is `ApiServer::handle_chat_completions` on the
server's default configuration. The smallest failing test enters THROUGH it, over
a synthetic in-memory checkpoint at tiny geometry with a generated tokenizer
fixture whose added tokens are `<|img|>` / `<|imgpad|>` / `<|endofimg|>`. A unit
test that constructs the tower by hand proves the class works, never that
anything reaches it.

The load-bearing case is **two DIFFERENT images, one prompt, compared on
LOGPROBS**. It is the only one that survives a tower replaced by a correctly
SHAPED constant: every text-only assertion — status 200, `prompt_tokens`,
`completion_tokens` — passes under that mutation, and the logprobs of the first
generated token do not.

#### 4.11.6 Evidence, measured 2026-09-01

Host: the developer's x86-64 Linux box, CPU queue, `-DVLLM_CPP_SERVER=ON
-DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`. No
GPU lease was taken and no number below is a performance number.

| Suite | Result |
|---|---|
| `test_dots3_note_vision` (new) | 8 cases, **4996 assertions**, 0 failed |
| `test_openai_api_server_dots3_mm_forward` (new) | 9 cases, **68 assertions**, 0 failed |
| `test_dots3_note_scaffold` | 26 cases, **110835 assertions**, 0 failed |
| `test_dots3_note_attn` | 51 cases, **6888 assertions**, 0 failed |
| `test_openai_api_server_mm_forward` (Qwen3-VL, untouched) | 9 cases, **73 assertions**, 0 failed |
| `test_model_registry` (repaired here) | 24 cases, **993 assertions**, 0 failed |
| **the FULL gate** — `ninja` all 1381 targets then `ctest -j 2` | **702/702, 0 failed**, 7 skipped for absent checkpoints, `NINJA_RC=0`, `CTEST_RC=0`, 158.06 s of `ctest` (rerun 2026-09-02 for the fresh-review repair) |
| `scripts/agent-preflight.sh` | rc 0 |
| `check-commit-style.py` / `check-commit-trailers.py` over `$(git merge-base origin/main HEAD)..HEAD` | rc 0 / rc 0 |

**The full gate found one thing reading did not.** `test_model_registry`'s
`registry_model_property` partitions every registration into hybrid,
multimodal-non-hybrid and text-only and asserts `supports_multimodal` per branch;
W5 had moved `Dots3NoteForCausalLM` into the text-only branch, and W6a's flag flip
made it red. It is repaired here, and its comment records the true -> false ->
true round trip rather than erasing it.

**The consistency measurement.** The tower against the independent
double-precision reference: `max |diff| 0.0533141 over a scale of 6.31441 =>
relative 8.44e-3`. The gate's bound is **0.02**, a 2.4x margin — wide enough that
a different libm or a different GEMM reduction order does not red it, tight
enough that M5 below (relative 5.9e-2) exceeds it by 2.95x. A bound at the round
0.05 a first draft carried would have cleared M5 by only 1.18x, which is one
compiler from a mute switch.

**WHAT THE 0.02 BOUND CANNOT SEE.** On a row with no oracle the honest statement
of what the gate does NOT detect is part of its evidence, not a caveat outside
it. Both numbers below were measured by the fresh review of #2523, on this
tree's own gate:

- **A uniform scale error passes until about 1.5%.** Multiplying the tower's
  output by 1.01 reads relative **0.0152** against the 0.02 bound, and the gate
  stays GREEN. The detection floor for a systematic MULTIPLICATIVE error is
  therefore ~1.5%: a missing or doubled scalar smaller than that is invisible
  here, and no assertion in the suite bounds the output's SCALE independently of
  its shape.
- **A named formula choice is below the gate's resolution.** Replacing the
  exact-erf GELU with the tanh approximation leaves the measurement
  BYTE-IDENTICAL to the baseline — the same printed digits, max |diff|
  **0.0533141** — because the bf16 store of `fc1` absorbs the whole difference.
  The gate cannot tell the two formulas apart at this geometry, so "we use
  upstream's GELU" is a claim the CODE and the upstream anchor carry, never one
  this measurement supports.

Neither weakens the two claims above: the 8.44e-3 agreement and M5's 5.9e-2 red
both stand. What they bound is the CLASS of defect the gate detects — a change
to the ORDER or the STRUCTURE of the arithmetic, which is what M5 is — and not a
small uniform rescale, and not a rounding-equivalent formula swap.

**Mutations.** Each was applied to the tree, REBUILT, and its test binary's
sha256 compared against the green baseline — a mutation that never reached the
binary reads as a passing test. Each file was then restored and `cmp` reported
byte-for-byte identity, and the rebuilt binaries hashed back to the EXACT green
baselines (`c6e83b90...` served, `b03f59e3...` tower).

**A CHANGED SHA IS NECESSARY AND NOT SUFFICIENT, and the argument below
originally overstated it.** The sha proves the build was not STALE, which is the
trap it was chosen for. It does not prove the mutation reached the code under
test: M4 changed the TOWER gate's binary sha purely by relinking a translation
unit that gate does not exercise, while that binary's behaviour was unchanged —
8/8, 4981 assertions, as the paragraph after the table records. The evidence
that a mutation was DETECTED is the CASE COUNT in the Result column, and nothing
else in this table can carry that weight.

| # | Mutation | Binary sha256 (served gate) | Result |
|---|---|---|---|
| — | green baseline | `c6e83b90359f1402…` | 7/7, 55 assertions |
| M1 | the `Dots3NoteVisionForward` call inside `encode_mm` DELETED | `936574cab156b69b…` | **RED** — 3 of 7 cases, 43 assertions reached |
| M2 | the tower REPLACED by a correctly-SHAPED constant | `5664578c254c8fa4…` | **RED — and only ONE case: "two DIFFERENT images give two different forwards".** 6 of 7 pass, including status 200, `prompt_tokens` and `completion_tokens`. This is the measurement behind §4.11.5: without the logprob case this mutation is invisible |
| M3 | the `.mm` read in `Dots3NoteModel::ForwardDevice` DELETED (`have_mm_embeds = false`) | `7285bd3e44221bcb…` | **RED** — the same single case, for the same reason: the vision rows never reach the residual stream |
| M4 | the production `MaterializeDots3NoteVision` call site in the LOADER deleted | `80901ec80645b598…` | **RED** — 3 of 7 cases |
| M5 | the per-head `q_norm`/`k_norm` moved from BEFORE the rope to AFTER it | tower gate `b39c3e36a0abbfad…` | **RED** — relative 5.9e-2 against the 2.0e-2 bound, a 7x jump from the green 8.4e-3 |

**THE REFUSAL REPAIR IS RED-FIRST, and its RED is the CASCADE rather than a
missing message.** The two served cases were written and built BEFORE the
refusal was widened, on binary `05848d1f4226e416…` (tower gate
`2d2dc86de602c004…`). Both failed at the install assertion — `kInstalled` where
`kRefusing` is required — so to show what that install then costs, the two
`REQUIRE`s were downgraded to `CHECK` in a scratch build (`34ff9a39094e0e18…`)
and the cases ran to the end. Verbatim, from that run:

```text
engine-fatal: EngineCore busy loop threw: vt: Dots3NoteForCausalLM encoder: the
  vision adapter emits 24-wide rows but the text tower is 16 wide
  (`adapter_out_dim`, vision.py:461 @ 9035151d6) at dots3_note_registry.cpp:202
async-llm: output handler saw engine death: EngineCore encountered an issue.
  CHECK( r.status == 400 ) is NOT correct!  values: CHECK( 500 == 400 )
  ...the TEXT request sent AFTERWARDS on the same server:
  {"error":{"code":500,"message":"EngineCore encountered an issue. ...
    [request submitted to a stopped AsyncLLM]"}}
  CHECK( t.status == 200 ) is NOT correct!  values: CHECK( 500 == 200 )
```

The merge-size case reaches the OTHER assert on the same path
(`dots3_note_registry.cpp:221`, "the tower produced 16 embedding rows for a
placeholder span of 4 tokens") and ends in the same
`request submitted to a stopped AsyncLLM`. The tower gate's own RED was 11
failed assertions over 4996 in 1 of 8 cases. The scratch file was restored and
`cmp` reported byte-for-byte identity before the fix was applied.

| # | State | Served-gate sha256 | Tower-gate sha256 | Result |
|---|---|---|---|---|
| — | new cases, refusal NOT widened | `05848d1f4226e416…` | `2d2dc86de602c004…` | **RED** — 2 of 9 cases; 11 of 4996 tower assertions |
| — | the same, `REQUIRE` -> `CHECK` so the cases run on | `34ff9a39094e0e18…` | — | **RED**, and the 500 + `stopped AsyncLLM` above is why |
| — | refusal widened (this repair) | `c8d9573ef3d605a7…` | `8f2b81436dcdac81…` | **GREEN** — 9/9, 68 assertions; 8/8, 4996 assertions |

**M4 leaves the TOWER GATE GREEN, and that is the point of having two files.**
`test_dots3_note_vision` materializes the tower itself, so deleting the
production call site does not move it: 8/8, 4981 assertions, on a tree where
nothing in the loader builds a vision tower at all. Only the served-request gate
sees it. That is AGENTS.md's "a unit test that constructs the type by hand proves
that the class works, never that anything reaches it", demonstrated rather than
argued.

**M5's first attempt did not compile** (`-Werror=unused-but-set-variable` on the
two tensor views the move orphaned) and the stale binary printed the GREEN
result. A build failure reading as a passing test is a recorded trap in this
tree; the row above is the SECOND attempt, whose `RC=0` and changed sha256 are
what make it evidence.

**One behaviour changed under measurement, and it is recorded rather than
smoothed over.** The first version refused a MoE tower only inside
`EncodeMmDots3NoteForCausalLM`. That throw happens in the engine's busy loop: it
stopped `AsyncLLM`, and every LATER request — TEXT ones included — came back 500.
The served-request gate caught it. The refusal now also runs in the chat
FACTORY, which turns it into a REFUSING seam: HTTP 400 naming the architecture
and the block, with the text path still answering afterwards. The encoder check
stays as defence in depth, on the same polarity Qwen3-VL's carries ("reaching
this point is a defect"). The gate asserts BOTH halves: 400 on the image, 200 on
a text request sent after it.

### 4.12 W6b computes the PYRAMID, and the released tower stops refusing

W6a shipped the dense blocks and refused the rest by name; this brick lifts that
refusal. The released `dots-studio/dots3-note-prev` carries 42 vision blocks of
which **17 are pyramid MoE**, so before W6b no image request against the real
checkpoint could be served at all. After it, `Dots3NoteVisionRefusal` returns ""
for that config and `MaterializeDots3NoteVision` loads all **2195**
`vision_encoder.*` tensors — W6a's 235 plus the 1960 the pyramid adds.

Issue [#2613](https://github.com/mudler/vllm.cpp/issues/2613). Upstream is
`vllm/models/dots3_note/nvidia/vision.py` read in `~/_git/vllm` at
**`9035151d6`**; the local clone's identity was re-asserted with `git rev-parse`
before a line was ported, and every anchor below names that SHA because
`dots3_note` does not exist at our parity pin `5559679229` and upstream has
already moved under this row.

#### 4.12.1 The router is not the language tower's, and the spelling proves it

| | vision (`MoESwiGLUFFN`) | language (`DeepseekV2MoE`) |
|---|---|---|
| router weight | `mlp.gate_weight` `[E_r, 1536]` BF16 | `mlp.gate.weight` |
| router bias | `mlp.router_bias` `[E_r]` **F32** | `mlp.gate.e_score_correction_bias` `[256]` F32 |
| experts | `mlp.experts.{e}.{fc1,fc2,fc3}` | `mlp.experts.{e}.{gate,up,down}_proj` |
| grouping | none | `n_group`/`topk_group` (both 1 here) |
| top-k | `min(int(capacity_factor), num_routed)` = 2 | `num_experts_per_tok` = 8 |
| combine | SELF-NORMALIZING (`/ (aggregated_gate + 1e-9)`) | plain weighted sum + shared expert |

Both spellings live in the SAME checkpoint, so conflating them is not a naming
preference: `EnumerateDots3NoteVisionTensors` claiming `mlp.gate.weight` on a
vision block would find no tensor and refuse the load for the wrong reason. The
gate asserts the two apart by name rather than assuming they differ.

The counts are the released index's own, read from the committed
`tests/vllm/models/fixtures/dots3_note_prev/index_full.json`: 17 routed blocks
carrying `4, 8, 12 ... 60, 64, 64` = **608** routed experts, `17 * 8 + 608 * 3 =
1960`, and `235 + 1960 = 2195`.

#### 4.12.2 The router IS `vt::MoeRouterTopK`, and that is a measurement

Upstream's router, line by line at `9035151d6`:

```
:180  gate_logits = F.linear(x_flat.float(), self.gate_weight.float())
:183  gating_prob = torch.sigmoid(gate_logits)                   # ELEMENTWISE
:192  gating_with_bias = gating_prob + router_bias.to(f32)
:193  _, topk_indices = torch.topk(gating_with_bias, k=topk, sorted=False)
:195  routed_weights = gating_prob.gather(1, topk_indices)       # UNBIASED
:196  if sigmoid and topk > 1: routed_weights /= (sum + 1e-9)
:200  routed_weights = routed_weights * self.router_scale
```

`vt::MoeRouterTopK` with `num_expert_group = 1` is that function and not an
approximation of it. Its grouped arm (`cpu_ops.cpp:2821-2948`, ported from
`grouped_topk_router.py:110-160`) computes sigmoid scores elementwise, adds the
bias to the SELECTION score only, reads the weight from the UNBIASED score,
renormalizes by the selected sum and then applies `routed_scaling_factor`. At one
group the group stage is definitionally inert — one group, `topk_group = 1`, the
mask all-ones — which is the same reasoning §4.10 records for the language
tower's `n_group == 1`.

**0 is not a smaller 1.** Passing `num_expert_group = 0` selects the op's
ungrouped path, which is SOFTMAX and ignores the bias entirely; the op wrapper
refuses a bias there for exactly that reason. So the choice of 1 is load-bearing
and is written with its reason beside it.

**The one difference, named rather than smoothed over.** Upstream divides by
`sum + 1e-9`; the shared op divides by `sum` under a `denom > 0` guard. That is
1e-9 RELATIVE against a bf16 store of 3.9e-3. The in-test reference spells
UPSTREAM'S version, so the difference is carried by the gate's tolerance and
measured, not defined away.

**A SECOND difference, found by the fresh implementer who finished this brick
rather than by the one who wrote it.** Upstream casts the routed weights to the
activation dtype before the combine — `routed_weights = (routed_weights *
self.router_scale).to(x_flat.dtype)` (`vision.py:200`) — so on a bf16 tower it
mixes the experts with BF16 coefficients and accumulates `aggregated_gate` from
those same rounded values. `vt::MoeRouterTopK` emits F32 weights and
`vt::MoeCombine` consumes F32, so this port mixes with f32 coefficients. This is
a WIDENING relative to upstream and `porting.md`'s memory-format rule asks for
the reason in writing: the buffer is `[L, top_k]` f32, four bytes per selected
slot, and its dtype is the shared ops' CONTRACT rather than a choice made here —
narrowing it would mean a bf16 round trip that no op in `include/vt/ops.h`
offers, written by hand beside the seam.

It is also self-cancelling to first order — but NOT by the symmetry an earlier
draft of this paragraph claimed, and the correction matters because that false
symmetry is what would have made the argument sound like a licence to narrow.
UPSTREAM does divide by the sum of the very coefficients it mixed with:
`aggregated_gate` is accumulated from the same rounded `routed_weights` values
that scaled the expert outputs (`vision.py:213`, `:215-217`), so its rounding
cancels term by term. **This port does not divide by that sum at all.**
`VisionMoeFfn` hands `vt::MoeCombine` the CONSTANT
`1.0f / (router_scale + 1e-9f)` (`dots3_note_vision.cpp`, the combine call),
never the realised sum of `tw`. The two denominators agree only because
§4.12.3's identity holds — after the renormalize at `:196-:199` the f32 weights
sum to `router_scale`, at f32 to about 1e-7 — and that identity is a property
`Dots3NoteVisionRefusal` defends, not an algebraic equality of the two
expressions. So what survives is upstream's ~4e-3 relative requantization of a
2-way mixture, mostly cancelled on upstream's own side, rather than a scale
error on either. The reference keeps double throughout and models NEITHER
rounding, which is why this sits inside the measured 1.01e-2 rather than beside
it. Recorded here because a reader comparing
`vision.py:200` with `VisionMoeFfn` will see the difference and is owed the
reason; it is not owed a brick.

#### 4.12.3 The combine's denominator is a CONSTANT, and that is why two arms refuse

Upstream's combine (`vision.py:202-217`) accumulates a per-token
`aggregated_gate` and returns `aggregated_output / (aggregated_gate + 1e-9)`.
After `:196-:200` the routed weights sum to `router_scale` for every token, so
that denominator is a per-tower CONSTANT and `vt::MoeCombine`'s single-float
`routed_scale` expresses it exactly as `1 / (router_scale + 1e-9)`.

That identity holds only where upstream renormalized. On the two arms where it
does not — `router_scoring_func == "softmax"`, and any block whose
`min(int(capacity_factor), num_routed)` is below 2 — the denominator is genuinely
per-token, and no op in `include/vt/ops.h` expresses a per-token scale on the
combine (`MoeCombine`'s is one float; `MulColVecF32` broadcasts over COLUMNS).
Serving them would mean either widening `vt::MoeCombine` — the op DeepSeek-V2's
SACRED token-exact path routes through — or a device sync per routed block to
renormalize a `[L, top_k]` f32 buffer on the host. Neither belongs in a brick
that is adding a tower, and no published dots3-note checkpoint selects either
arm: the released config and `DotsMoEVitConfig`'s own defaults are both sigmoid
at capacity 2. **Both are refused BY NAME, listed under `## Owed`, and owned by
[#2615](https://github.com/mudler/vllm.cpp/issues/2615).**

#### 4.12.4 Where the shared seams carry it, and the one place W6a's choice is wrong

Every routed expert's SwiGLU rides `layers::MlpGateUpMethodBase`, exactly as the
dense blocks do — but through `UnquantizedMlpGateUpSplitMethod` rather than
W6a's merged `UnquantizedMlpGateUpMethod`, and the reason is measured. Merging
`fc1|fc3` needs a COPY out of the mmap, which on the released checkpoint is
`608 experts x 2 x 2112 x 1536 x 2 B` = **7.9 GiB** of resident bytes bought for
one fewer kernel launch. The split method exists in `linear.h` for precisely this
case and says so in its own prose. The DENSE blocks keep the merged operand,
where the same copy is 649 MiB across 25 blocks and was already paid by W6a.

`vt::MoeRouterTopK` and `vt::MoeCombine` carry the router and the combine. The
gather/scatter around the experts mirrors `dots3_note_device.cpp`'s reference MoE
arm — the arm the CPU queue takes there too. What is NOT hoisted is that file's
grouped-GEMM fast path: it reads a Matmul-B expert layout this tower does not
have, and adding one would be a residency change on a brick with no performance
claim.

#### 4.12.5 The F32 is on the OUTPUT, not on the operands

Upstream writes `.float()` on both sides of the router GEMM (`:180`). Both sides
are bf16-VALUED — `x` is the bf16 `norm_2` output and `gate_weight` is BF16 on
disk in the released index — and a bf16 x bf16 product is exact in f32. A
bf16-operand GEMM with an f32 accumulator therefore IS
`F.linear(x.float(), w.float())`, while widening the stored operand would double
the resident bytes for no information at all. So the logits buffer is f32 and the
operands are not, with the reason written beside it — `porting.md`'s
memory-format rule applied in the direction it is usually not.

`router_bias` is the exception that really is f32, and it is UPSTREAM'S choice:
`register_buffer("router_bias", torch.zeros(num_routed, dtype=torch.float32))`
(`vision.py:152-154`). Those 17 buffers are exactly the 17 F32 tensors the
released vision tower carries against 2178 BF16 ones, and the census that says so
is §4.4's. The loader asserts the dtype in BOTH directions — `RequireVisionShape`
refuses a widened weight, `RequireF32VisionShape` refuses a narrowed bias — and
the gate carries a case that writes the bias BF16 and reads the refusal, which is
this row's W2 F1 fixture row pointed at the vision router.

#### 4.12.6 The gate needs a shape the dense arm did not

**Top-k selection is a DISCRETE choice, so its error is bimodal.** Either the
same experts were chosen and the output error is the ordinary bf16 one, or a
different expert was chosen and the output is a different function. There is
nothing in between for a relative bound to measure, and a selection defect that
happens not to flip on the fixture leaves a tolerance green while saying nothing.
A tolerance alone on this path is a mute switch.

So the gate does three things a tolerance cannot:

1. **Selection-SET equality, per token**, against the reference's own
   independent scan. A set rather than a sequence, because
   `torch.topk(..., sorted=False)` leaves the ORDER unspecified upstream and the
   combine is a sum.
2. **The minimum decision MARGIN, printed** — the gap between the last selected
   and the best rejected biased score, minimised over tokens — so the reader
   knows how much room the assertion had. A margin at zero would mean the fixture
   decides its routing by a tie and the agreement is luck.
3. **The instrument's own precondition**, asserted on two axes: EVERY routed
   expert must be selected by some token (a per-expert load floor, not a count
   of distinct ids), and more than two of the `C(4,2) = 6` possible pairs must
   occur. Without both, a router that ignored its input and sent every token to
   the same pair would pass the set assertion — and the review that repaired
   this section found the fixture doing very nearly that: 13 of 16 tokens on one
   pair, expert 0 never selected, and the old `distinct >= 3` bound met with
   zero slack. §4.12.9 carries the reseed and the before/after spread.

**And a fourth thing it does NOT do, named here because the reader will
otherwise assume the served suite covers it.** The three assertions above are
about SELECTION. A routed-path defect that preserves the selection set — M6 in
§4.12.9, which combines each expert's output with the OTHER selected expert's
weight — is caught by this gate on TOLERANCE ALONE, and by the served suite not
at all. §4.12.9 also records why no cheap served case can be made to catch it:
every served assertion available here is "two answers differ", and a
deterministic arithmetic defect leaves both answers well-defined and distinct.
That was measured with an all-routed fixture, not assumed.

The reference (`namespace ref` in `tests/vllm/models/test_dots3_note_vision.cpp`)
shares NO helper with the implementation: its own sigmoid, its own selection
scan, its own per-expert SwiGLU, and the literal `aggregated_gate` division
rather than the constant the implementation folds into `routed_scale`. Keeping
the literal form is what makes the 1e-9 difference between them a measurement.

**This is still a CONSISTENCY gate and it says so.** §6.4 records option B: the
checkpoint is 298.67 GB fp8 / 576.89 GB bf16 against 119-122 GiB hosts, so vLLM
cannot be run on it on any hardware this project owns and no denominator exists.
Two implementations agreeing is not either of them being right, and **no
performance number is claimable on any axis** while B holds.

#### 4.12.7 The config arms W6a deferred: four lifted, one still refused

| Arm | W6b | Why |
|---|---|---|
| `adapter_type = pixel_shuffle_mlp` | **LIFTED** | `PixelShuffleAdapter` (`vision.py:419-461`) is a real published adapter with its own state dict (`proj.0`/`proj.1`/`proj.3`) and its own token order. Implemented, and gated against a reference that spells the reshape/permute chain rather than the closed form the implementation gathers by |
| `post_norm = false` | **LIFTED** | W6a's forward and enumerator already had the branch; only the refusal stood in front of it. Upstream creates the `post_trunk_norm` attribute only under `if config.post_norm:` (`vision.py:525-526`) and guards the call with the same flag in `forward` (`:673-674`) — there is no `nn.Identity` in the file, so the step is ABSENT rather than an identity; one tensor fewer |
| `use_qk_norm = false` | **LIFTED** | upstream builds no `q_norm`/`k_norm` module (`vision_attention.py:145-147`) and the checkpoint ships neither; two tensors fewer per block |
| `is_causal = true` | **LIFTED** | `causal=self.is_causal` on the FLASH family (`vision_attention.py:265`, `:291`, `:302`), which is what the released `attn_implementation = flash_attention_3` selects |
| `use_bias = true` | **STILL REFUSED**, [#2616](https://github.com/mudler/vllm.cpp/issues/2616) | see below |

**`is_causal` needed a decision, and the record has to carry it.** The two EAGER
attention classes store `self.is_causal` and never read it:
`VisionAttention.forward` (`:172-204`) builds its mask from `cu_seqlens` alone and
`VisionAttentionV2.forward` (`:210-239`) takes a plain full softmax per segment.
So upstream silently ignores a true `is_causal` on an eager implementation and
masks on a flash one. This port follows the FLASH arm, because that is what the
released `vision_config` asks for. On the released config the flag is false and
the two arms coincide, so the choice becomes visible only on a checkpoint that
sets it — which is why it is written down here rather than left to be found.

**`use_bias` is refused for three reasons and none of them is effort.** No
published dots3-note checkpoint sets it (`DotsMoEVitConfig`'s own default is
`False`, `vision.py:43`, and the released config agrees). The shared
`layers::MlpGateUpMethodBase` seam has no bias operand, so lifting it means
either extending the seam every model in the tree routes its MLP through, for a
configuration none of them has, or writing the two GEMMs by hand beside it —
which is the parallel path AGENTS.md forbids. And it would land UNREACHED: the
only production entry point that could reach a `use_bias` arm is a checkpoint
that declares it, and the only such checkpoint would be a fixture written to
reach it. The refusal names the keys, the seam and the issue, and a served
request against such a checkpoint gets HTTP 400 with the text path still
answering afterwards.

**One new refusal that is not a deferral.** `pixel_shuffle_mlp` at
`adapter_merge_size != 2` is refused because `_pixel_shuffle` hard-codes
`scale_factor=0.5` (`vision.py:401-416`, `:456`) while `merged_dim = in_dim *
merge_size**2` (`:431`) sizes `proj.0` by the key — at anything but 2 the two
widths disagree and upstream raises on the shape. Similarly, an ODD grid side
under that adapter is refused in the forward: `_pixel_shuffle` duplicates the
first row or column (`:402-405`), so it emits `ceil(h/2)*ceil(w/2)` rows while
the prompt expands `prod(grid) // merge**2` placeholders
(`multimodal.py:151-155`). Upstream's two halves disagree there and no such
request is servable by either.

#### 4.12.8 Reachability

The production entry point is unchanged: `ApiServer::handle_chat_completions` on
the server's default configuration. `test_openai_api_server_dots3_mm_forward`
serves a checkpoint whose block 1 is a 4-expert pyramid block through the whole
chain — chat seam, placeholder expansion, `AsyncLLM`, scheduler, encoder
admission, `GPUModelRunner::execute_model`, `EncodeMmDots3NoteForCausalLM` ->
`Dots3NoteVisionForward` -> `EmbedMmDots3NoteForCausalLM` -> `ForwardDevice`'s
`inputs_embeds` arm — and asserts the TWO-DIFFERENT-IMAGES LOGPROB case, which
is the only assertion of that file that survives a tower replaced by a correctly
shaped constant.

`test_dots3_note_vision` measures the arithmetic and would pass on a tree where
nothing calls the tower, because it materializes the weights itself. That is the
division AGENTS.md's "Nothing lands dead" asks for, and the M5 mutation below
demonstrates it rather than asserting it.

#### 4.12.9 Evidence, measured 2026-09-03

Host: the developer's x86-64 Linux box, CPU queue, `-DVLLM_CPP_SERVER=ON
-DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`, `-j
2`. No GPU lease was taken and **no number below is a performance number** —
§6.4 option B holds, so none is claimable on any axis.

**Oracle identity, asserted rather than assumed.** `git rev-parse` in
`~/_git/vllm` reports the clone detached at
`5559679229bc961848b121ccdeaa8fa5d79bec98`, this project's parity pin, and
`ls vllm/models/dots3_note/nvidia/` fails there: **the pin contains no dots3-note
at all.** Every anchor in §4.12 therefore names
`9035151d6c9fb726181469f9e6aa9ccbf9a5dacb` — "[Model] Add native Dots3 NOTE
multimodal support (#51255)" — read out of the same clone's object store with
`git show 9035151d6:vllm/models/dots3_note/nvidia/vision.py`.

| Suite | Result |
|---|---|
| `test_dots3_note_vision` | 12 cases, **20798 assertions**, 0 failed |
| `test_openai_api_server_dots3_mm_forward` | 12 cases, **177 assertions**, 0 failed |
| `test_dots3_note_scaffold` | 26 cases, **110835 assertions**, 0 failed |
| `test_dots3_note_attn` | 51 cases, **6888 assertions**, 0 failed |
| **the FULL gate** on the merged head — `cmake --build build -j 2` then `ctest -j 2` | **709/709 targets, `NINJA_RC=0`**; **720/720 tests passed, 0 failed**, 7 skipped for absent checkpoints or an absent CUDA device, `CTEST_RC=0`, 144.95 s |
| `scripts/agent-preflight.sh` | rc 0, 25 record gates ok |
| `check-commit-style.py` / `check-commit-trailers.py`, `--range $(git merge-base origin/main HEAD)..HEAD` | rc 0 / rc 0 |

W6a's suites were 8 cases / 4996 assertions and 9 cases / 68 assertions; the
pyramid adds 4 tower cases and 3 served ones.

**The reference is independent, and that was re-tested rather than inherited.**
`namespace ref` was extracted and every qualified name inside it enumerated with
comments and string literals stripped: **105 occurrences of 13 distinct
`std::` names, and nothing else** — `array`, `cos`, `erf`, `exp`, `max`, `min`,
`pow`, `sin`, `sort`, `sqrt`, `string`, `to_string`, `vector`. Thirteen is
counted here rather than left to the reader, because the review that repaired
this section found the count restated as 14 in the report that accompanied it.
Not one `vllm::` and not one `vt::`. The single
textual `vt::MoeCombine` in that span is inside a COMMENT explaining why the
reference does the self-normalizing divide literally instead of folding it. The
only non-`std` symbols it reaches are `dots3_tiny::TinySpec` and
`dots3_tiny::TinyCheckpoint`, which are the FIXTURE that writes the checkpoint
both sides read, not the implementation. So the agreement below is two
implementations agreeing, which is what §6.4 option B buys and is not either of
them being shown to match vLLM.

**The consistency measurements**, as the suite prints them:

| Arm | relative deviation | bound |
|---|---|---|
| dense tower (W6a, unchanged) | `max \|diff\| 0.0533141 / scale 6.31441` = **8.44e-3** | 0.02 |
| **pyramid tower** | `max \|diff\| 0.0612129 / scale 6.04501` = **1.01e-2** | 0.02 |
| `post_norm = false` | **6.59e-3** | 0.02 |
| `use_qk_norm = false` | **8.36e-3** | 0.032 |
| `is_causal = true` | **1.53e-2** | 0.032 |
| `pixel_shuffle_mlp` | **1.31e-2** | 0.02 |

Every routed row moved when the review repair reseeded the router (below); the
dense row did NOT, which is the check that the reseed touched the routed block
and nothing else. The pyramid's 1.01e-2 is the same order as the dense arm's
8.44e-3, which is what one expects when the SELECTION agrees and the only
difference left is bf16 storage. A routed block is also a different function
from a dense one on this fixture by `max |diff| 9.25`, so that agreement is not
an accident of a branch that did not matter.

**The discrete assertion, its population, and the margin it had — REPAIRED
after review, and the repair is the interesting part.**

As first landed this fixture was at the floor of its own precondition. Over 16
tokens: 13 routed to `{1, 2}` and 3 to `{2, 3}`, so expert 2 sat in EVERY set,
**expert 0 was never selected at all**, only 2 of the 6 possible pairs occurred,
three tokens carried the whole discriminating population of the set assertion,
and `distinct >= 3` passed against a spread of exactly 3 — zero slack. The gate
still bit: M2 forces every id to `{0, 0}`, which differs from `{1, 2}` and
`{2, 3}` alike, so all 16 set assertions had to fire and the recorded M2 run
did. But that says only that the crudest selection defect is visible from
anywhere; the strongest assertion in the file was standing on the weakest
arrangement of the fixture, and expert 0 is the index an off-by-one lands on.

`TinySpec::v_router_seed_nudge` was added for this and nothing else. It offsets
the two seeds that draw `mlp.gate_weight` and `mlp.router_bias`, the shared
`next()` stream still advances once per tensor, and every other tensor in the
checkpoint is byte-identical — which the unmoved dense row in the table above
confirms independently. The 64 offsets `0..63` were swept on the tower this
fixture actually builds; **42 is the only one that reaches all four experts AND
all six pairs**, and the search is reproducible from the fixture alone.

| | as landed | at nudge 42 |
|---|---|---|
| experts selected | 3 of 4 (**expert 0 never**) | **4 of 4** |
| per-expert load over 32 slots | 0 / 13 / 16 / 3 | **7 / 4 / 9 / 12** |
| distinct selection SETS | 2 of the 6 pairs | **6 of the 6 pairs** |
| minimum decision margin | 4.0077e-3 at token 1 | **1.25999e-2 at token 8** |
| pyramid relative deviation | 7.81e-3 | 1.01e-2 |

The precondition was strengthened with the fixture rather than after it: it now
asserts a per-expert load FLOOR (every routed expert selected by some token) and
a distinct-SET count above two, instead of counting distinct ids against a bound
with no slack. The `CHECK_MESSAGE` that reports a set mismatch no longer indexes
slots `[0]` and `[1]` by hand while `top_k == 2` is only a `CHECK`.

**The margin got THREE TIMES LOOSER and that is a cost, not a win.** The
implementation's logits come from a bf16-operand GEMM with an f32 accumulator
over a 16-wide reduction, so they sit within ~1e-3 relative of the reference's
double ones; through a sigmoid, whose slope is at most 1/4, that is ~2.5e-4 of
score. 1.26e-2 is ~50x that where 4.01e-3 was ~16x. A SMALL margin is the useful
direction — it means the fixture sits near the decision boundary, so a selection
defect has somewhere to show — and the trade was taken anyway, because the old
margin was bought by routing 13 of 16 tokens to one pair and never exercising
expert 0. A tight margin over a population that cannot discriminate is not a
sharper instrument.

**Mutations.** Each was applied, REBUILT with `BUILD_RC` read before anything
else, its two test binaries hashed, both suites run, then the file restored and
`git diff --quiet` used to prove byte-for-byte identity. **The whole table below
was RE-RUN on 2026-09-03 after the fixture reseed above**, because a mutation
table measured against a different fixture is a record of a tree that no longer
exists. The green baselines are `78a0bb3bda3a1f96…` (tower) and
`ebc83d5b15ab857a…` (served), and after the last restore both rebuilt binaries
hashed back to EXACTLY those.

**A changed sha is necessary and not sufficient; the CASE COUNT is the evidence.**
This run proved it the hard way. The first attempt at M3 did not COMPILE —
`-Werror=unused-but-set-variable` on the now-unused `rbias` — and the harness
went on to run the M2 binaries again under an M3 label, reproducing M2's shas
digit for digit and M2's red assertion for assertion. It read as a convincing
M3. `BUILD_RC` is what caught it, and M3 below is the rerun that compiles.

| # | Mutation | tower sha256 | served sha256 | Result |
|---|---|---|---|---|
| — | green baseline | `78a0bb3bda3a1f96…` | `ebc83d5b15ab857a…` | tower **12/12, 20798**; served **12/12, 177** |
| M1 | the MoE branch DELETED in the forward, so a routed block falls back to dense (`if (bw.is_moe)` -> `if (false)`) | `5d21d6ff21af1848…` | `571762a22c9b2079…` | **RED both** — tower 9/12, 3 cases THREW `resident weight: EMPTY tensor has no host bytes to alias`; served 9/12, 3 of 167 assertions |
| M2 | every top-k id forced to expert 0 after the download, so the CAPTURE sees it too (a selection defect with valid shapes) | `ab5ac5a2544b00d4…` | `63ac10a5712517c6…` | **RED both** — tower 10/12, **27 assertions**, itemised below; served 11/12, the router-bias case |
| M3 | `router_bias` dropped from the gating (`&rbias` -> `nullptr`, with `(void)rbias`) | `6a848ffc66ea0386…` | `b58330b88b6336f4…` | **RED both** — tower 10/12, **9 assertions**: 3 SET (tokens 2, 3, 8), `agreed == L`, and 5 tolerance (0.161, 0.155, 0.150, 0.0566, 0.180); served 11/12, the router-bias case |
| M4 | the routed FFN output replaced by a correctly-shaped constant (every expert skipped, `expert_out` left zero) | `5c3cc4f0f82635fe…` | `bef9055aabbabb53…` | **RED both** — tower 10/12, 5 tolerance assertions (0.202, 0.267, 0.282, 0.500, 0.212 against 0.02/0.032), **zero SET**; served 11/12 |
| M5 | the production call site `w.vision = MaterializeDots3NoteVision(shards, w.vision_params)` DELETED (`dots3_note.cpp`) | `347b35ebf166e2ae…` | `3de555b3fdbb819e…` | tower **12/12 GREEN, 20798**; served **RED 6/12**, 6 of 155 assertions |
| **M6** | **the routed slot SWAPPED in the scatter** — `tj.first * k + tj.second` -> `tj.first * k + (k - 1 - tj.second)`, so every expert's output is combined with the OTHER selected expert's routing weight | `48389c5cb530a22a…` | `d0705c9b24329847…` | tower **RED 10/12, 5 assertions, ALL tolerance (0.283, 0.307, 0.301, 0.0748, 0.312), ZERO SET**; served **12/12 GREEN, 177 assertions** |

**M2's 27, itemised**, because "a changed count" is not the evidence and the
previous version of this row said "23 assertions … all at the SET assertion",
which was wrong on both halves: 23 was the TOTAL, and the set assertions were
16 of it — the remaining 7 being `agreed == L`, the old `distinct >= 3`, and 5
tolerance reds. At nudge 42 the total is 27 and it decomposes as **16 SET**, 1
`agreed == L`, 1
`distinct.size() == num_routed`, **3 `load[e] >= 1`** — experts 1, 2 and 3 go
unrouted when everything is forced to expert 0, which is the new precondition
firing — 1 `distinct_sets.size() > 2`, and 5 tolerance.

**M6 is the sharpest mutation anyone has run on this path, and it is the one the
served suite cannot see.** It preserves the selection SET exactly — the routed
ids, the reported spread (4 of 4 experts, loads 7/4/9/12, 6 of 6 pairs) and the
reported margin (1.25999e-2 at token 8) are bit-identical to the green run — and
it still multiplies every expert output by the wrong one of the two routing
weights. That is a genuine routed-path arithmetic defect. The tower gate catches
it on TOLERANCE ALONE — 0.283, 0.307 and 0.312 against a 0.02 bound, 0.301 and
0.0748 against 0.032, so 2.3x to 15.6x — and by nothing else. The served suite
does not catch it at all.

**M5 is the reachability measurement and it says two things.** The served suite
loses half its cases when the materialization call site goes, so the pyramid is
reached from `ApiServer::handle_chat_completions` on the default configuration
rather than only from a test that builds the type by hand. And the tower suite
stays fully green, so that suite measures ARITHMETIC and never reachability —
which is the division `.agents/reachability.md` asks a change to demonstrate
rather than assert.

**WHAT THIS GATE CANNOT SEE**, extending §4.11.6 rather than restating it. W6a
measured two limits that still hold unchanged: a uniform MULTIPLICATIVE error
passes until about 1.5%, and swapping the exact-erf GELU for the tanh
approximation is below the gate's resolution. W6b adds a third, and it is about
the served side:

- **The two-different-images LOGPROB case does NOT detect a router defect.** M2,
  M3 and M4 are three different ways to break the routing, and all three left
  that case GREEN. The reason it survives them is that two different images
  still produce two different logprobs however wrong the routed block is. That
  case is the load-bearing assertion for the TOWER being reached, and it is not
  an assertion about the ROUTER. The served case that does catch all three is
  "the router BIAS changes what the server answers", which compares two
  checkpoints differing in `mlp.router_bias` and in nothing else — a premise the
  case asserts by diffing all the fixture's tensors and requiring exactly one to
  differ. It exists because of this measurement, not before it.

- **NO served case detects a routed-ARITHMETIC defect that preserves the
  selection set, and the whole served suite is green under one.** M6 is that
  defect. The bias case catches M2, M3 and M4 because each of them changes WHICH
  expert runs, so the two checkpoints stop disagreeing about the selection and
  the two answers collapse together. M6 changes none of that: the selection is
  identical, the bias still moves it, and the two answers still differ — they
  are simply both wrong. `test_openai_api_server_dots3_mm_forward` reports
  **12/12, 177 assertions, 0 failed** with M6 applied. So this file gates
  REACHABILITY and BIAS-DEPENDENCE of the routed block and nothing about the
  arithmetic inside it; `test_dots3_note_vision` is where routed arithmetic is
  caught, and there it is caught by tolerance rather than by the discrete
  assertion.

- **COULD a served case gate routed arithmetic? Measured, and the answer is no
  for every cheap shape of it.** The obvious candidate is a fixture whose block
  0 is ROUTED rather than dense, so that no dense block stands between the
  pixels and the answer. It was BUILT and RUN rather than argued: a temporary
  served case at `v_pyramid = {4, 4}`, two different images compared on
  logprobs, **passes with M6 applied** (its two logprobs both moved — ` `/`w`
  became ` `/`<|img|>` — and they still differ from each other), and passes
  without it. The dense block 0 was never the reason. The reason is the SHAPE of
  the assertion: every cheap served assertion available here is "two answers
  differ", and a deterministic arithmetic defect that leaves both answers
  well-defined and distinct is invisible to a difference. What would catch M6 is
  a served assertion on a VALUE, and there are only two sources for one. A
  hard-coded golden logprob is a cross-platform constant with no oracle behind
  it (§6.4 option B), whose tolerance would have to be tighter than the bf16
  spread across the runners this repository builds on, which nothing here has
  measured. Driving the in-test double reference through the whole chain —
  tower, adapter, TEXT tower and sampler — and comparing the served logprob to
  it would work, and it is a strictly larger duplicate of
  `test_dots3_note_vision` plus a text-tower reference this row does not have.
  Neither is a served case; both are the tower gate wearing a server. **The
  division stands as it is, and it is now written down rather than implied.**

### 4.13 W6c ports PIL's resampler, and the served image no longer has to be a multiple of 28

**The gap W6c closes.** W6a landed the image processor and W6b made the released
tower compute, so after W6b every part of the served image chain worked except
its first step. `Dots3NoteImageProcessor::ProcessImage` computed
`Dots3NoteResizedSize` correctly and then REFUSED whenever that size differed
from the size it was handed, because the pixel RESAMPLING behind
`image.resize(..., Image.Resampling.BICUBIC)` (`common/processor.py:174` @
`9035151d6`) was not ported. `factor = patch_size * merge_size` is 28 on
`dots-studio/dots3-note-prev`, and upstream ALWAYS resizes, so the refusal
turned away essentially every real photograph and screenshot. It landed as an
HTTP 400 at `ApiServer::handle_chat_completions` with both sizes in the message,
so it was never a silent skip — it was the last thing standing between the
server and a real image. Issue
[#2537](https://github.com/mudler/vllm.cpp/issues/2537).

**What was already ported, and what W6c leaves alone.** The GEOMETRY is W6a's
and W6c does not touch it: `RoundByFactor` (Python `round`, half-to-EVEN, via
`std::nearbyint` and deliberately not `std::round`), `CeilByFactor`,
`FloorByFactor` at `dots3_note_processor.cpp:30-45`, and `Dots3NoteResizedSize`
mirroring `resized_size` (`common/processor.py:97-146` @ `9035151d6`) including
the `min_pixels` floor, the `max_pixels` ceiling and the second rebalance when
the floor pushes the size back past the ceiling. That clamp is already honoured
from `preprocessor_config.json`, and §4.11's gate already measures it.

**What `Dots3NoteResizedSize` still ignores, and why that is not W6c's.**
Upstream's `resized_size` also reads a per-call `detail` string, a
`self.image_details[detail]` override table, and explicit `target_height` /
`target_width` arguments (`common/processor.py:97-119`). None of the three
reaches this port, because the OpenAI chat seam this row owns never supplies
them: `RouteDots3NoteImageRgb` calls `ProcessImage(rgb, height, width)` with the
decoded image's own size and nothing else, and `image_details` is a
`Dots3NoteProcessor` constructor argument no released `preprocessor_config.json`
carries. Wiring `detail` through is a FRONT-END change to the chat seam's
request parsing, which is W8's, and it stays owed under `## Owed` beside the
other W8 items rather than being widened into here. This is stated so a reader
does not read "the budget clamp is honoured" as "every argument of
`resized_size` is honoured".

#### 4.13.1 `Image.Resampling.BICUBIC` is not "a bicubic kernel"

The one thing that could quietly make every served image wrong is treating
PIL's `resize` as a four-tap cubic interpolation. It is not one, and on a
DOWNSCALE — which is what a chat request almost always asks for — the difference
is not a rounding difference. The algorithm is `ImagingResampleInner` and its
`8bpc` horizontal/vertical passes in Pillow's `src/libImaging/Resample.c`, read
at tag `12.1.1`, which is the version installed on this developer host and the
version `import PIL` reports there. Its parts, in the order they run, are:

1. **The filter.** `bicubic_filter` with `a = -0.5` and `support = 2.0`:
   `((a+2)x - (a+3))x² + 1` for `x < 1`, `(((x-5)x + 8)x - 4)a` for `x < 2`, zero
   beyond. This half is the textbook Keys kernel, and it is the only half a
   textbook gives you.
2. **Support SCALING on downscale.** `precompute_coeffs` sets
   `scale = in/out` and `filterscale = max(1.0, scale)`, then
   `support = 2.0 * filterscale` and `ksize = ceil(support)*2 + 1`. Every filter
   argument is divided by `filterscale`. A 5x downscale therefore reads a
   21-tap window and behaves as a weighted AREA AVERAGE, not as a 4-tap
   interpolation. A plain 4-tap bicubic aliases instead, and the error is
   structural rather than small.
3. **The sample centre.** `center = in0 + (xx + 0.5) * scale` with the box
   `in0 = 0`, `in1 = inSize`; the weight of input column `x + xmin` is
   `filter((x + xmin - center + 0.5) / filterscale)`. The two halves of that
   `+0.5` are the classic half-pixel trap: dropping either one shifts the whole
   image by half an output pixel and leaves every shape valid.
4. **The window.** `xmin = (int)(center - support + 0.5)` clamped at 0,
   `xmax = (int)(center + support + 0.5)` clamped at `inSize`, then
   `xmax -= xmin` so it is a COUNT. The cast truncates toward zero; below zero
   the clamp makes truncation and floor agree, and the upper bound is never
   negative, so `(int)` is faithful and not a simplification.
5. **Per-output NORMALIZATION.** The weights are divided by their own sum,
   which is what makes the clipped edge windows and the widened downscale
   windows preserve brightness. Skipping it darkens the borders and, on a
   downscale, scales the whole image — a uniform shift a relative tolerance can
   absorb.
6. **Fixed point.** `normalize_coeffs_8bpc` converts each weight to
   `int32` as `(int)(±0.5 + w * 2²²)` (`PRECISION_BITS = 32 - 8 - 2 = 22`),
   which is round-half-away-from-zero. Each output accumulates from
   `1 << 21` — the rounding half — and `clip8` returns
   `clamp(acc >> 22, 0, 255)` with an ARITHMETIC shift, so the rounding is
   round-half-up and the clamp is saturating.
7. **Two passes over a uint8 INTERMEDIATE.** Horizontal first, into a real
   8-bit image; then vertical over that. The intermediate is quantized, so the
   result is NOT the separable float computation rounded once. Each pass is
   skipped entirely when that axis does not change size, which for
   `box = (0,0,w,h)` is exactly `out == in` on that axis.

**The one deviation, and why it is an identity.** PIL computes the horizontal
pass only over source rows `[ybox_first, ybox_last)` — the rows the vertical
pass will read — and shifts `bounds_vert` to match. `PilResizeBicubicRgb`
computes all `in_h` rows and indexes the intermediate absolutely. The horizontal
pass reads exactly one source row per output row and writes exactly one
intermediate row, so the two agree byte for byte on every row either of them
computes; the deviation only costs the rows outside the window. It is recorded
here because "we skipped an optimization" and "we changed the maths" look the
same in a diff.

**The one widening, and its bound.** PIL accumulates in `int`. This port
accumulates in `int64_t`. The weights are normalized to sum 1, which does NOT
bound their magnitudes at 1: `1.25` is the four-tap phase-0.5 value
(`2*0.5625 + 2*0.0625`), and once `filterscale > 1` the raw weights no longer
sum to 1, so dividing a truncated window by a small row sum amplifies them past
it. Swept over every `(in, out)` in `1..800 × 1..800` (640,000 pairs), the
supremum of `sum |k|` is **1.268771**, at `in = 208`, `out = 193`, output index
96 — an INTERIOR window — with normalized taps
`(-0.067193, +0.567193, +0.567193, -0.067193)`. Taking `sum |k| <= 1.27`,
`|acc| <= 255 * 1.27 * 2²² + 2²¹`, about `1.36e9`, inside `int32`'s `2.15e9`;
PIL itself relies on a tighter bound still, because its `clip8` lookup table
only covers `acc >> 22` in `[-640, 639]`, and the worst index the sweep produces
is 324. The wider accumulator therefore removes an overflow that cannot occur
rather than changing an answer. The conclusion is unchanged from the `1.25` this
paragraph first carried; the constant was wrong and the margin absorbed it.

**This is PIL's resampler and NOT torchvision's, and the two must not be
confused.** `qwen3vl_processor.cpp` refuses its own resize (`:110-116`,
`:271-274`) and names torchvision bicubic, which is a different algorithm:
`antialias=False` by default on tensors, no support scaling, and float output
with no uint8 round trip. The new file is named for PIL for that reason, and its
header says so, so that a later reader does not discharge the Qwen3-VL debt with
it.

#### 4.13.2 Where it lands

`include/vllm/multimodal/pil_resize.h` + `src/vllm/multimodal/pil_resize.cpp`
carry `PilResizeBicubicRgb(src, in_h, in_w, out_h, out_w)` over HWC uint8 RGB.
`Dots3NoteImageProcessor::ProcessImage` deletes the throw and calls it whenever
`Dots3NoteResizedSize` moves either side, then patchifies the resized buffer at
the resized stride. The `min(h,w) < factor/4` and aspect-ratio refusals stay:
they are upstream's own `ValueError`s and not a gap.

The RGBA compositing arm at `common/processor.py:157-162` (white background,
alpha as mask) and the non-RGB `convert("RGB")` arm are NOT reached here: the
chat seam's `ImageCodecFn` hands `ProcessImage` three-channel RGB already, so
the mode normalization happens in the codec, upstream of this function. That is
a statement about who owns it, not a claim that it is done — it stays with the
codec, which is W8's.

#### 4.13.3 The gate, and the four shapes a tolerance cannot see

§6.4 option B still holds: there is no oracle for this row, so the gate is an
independent in-test reference sharing no helper with the implementation. For a
RESAMPLER a mean-error tolerance is a particularly weak instrument, so the gate
asserts these separately:

- **Per-pixel MAXIMUM error, in units of one 0-255 level**, not a mean. The
  fixed-point reference reproduces steps 1-7 above and the bound is EXACT
  equality — max error 0 levels — because both compute the same integer
  quantization. A second, purely double-precision reference that stops before
  step 6 is asserted at **2 levels**, the bound the test carries; the DERIVED
  ceiling is 1.13439 (§4.13.4) and the measured value 1.07588.
- **What the two arms do and do not cross-check.** They are independent of the
  IMPLEMENTATION. They are NOT independent of each other: `ResizeContinuous`
  calls `ref_resample::Weights`, the same function `ResizeExact` calls, and the
  two arms differ by step 6 alone. So the pair cross-checks the QUANTIZATION —
  a shared misreading of steps 6-7 moves the continuous arm — and supplies no
  independence on the GEOMETRY. Concretely: had the centre been read as
  `xx * scale` with no `+ 0.5` and written into both `PrecomputeCoeffs` and
  `Weights`, every case would report `maxe == 0` and `maxc == 0.0`, both CHECKs
  green and the served suite green, with every image shifted half an output
  pixel. Steps 1-5 therefore rest on ONE reading of `Resample.c` plus the Pillow
  cross-check below, which is evidence and not a gate. That is the honest
  position, and it is why the Pillow number is recorded at all.
- **A NON-SQUARE image with a non-square target.** A transposed loop or an
  axis swap is invisible on a square test image and this row's existing
  fixture image is 8x8.
- **A half-pixel probe.** A ramp image resampled at a scale where the
  centre offset is decisive, asserted against the reference exactly; the
  mutation table below shows the offset mutation moves it.
- **Both extremes.** A hard 0/255 edge downscaled, where bicubic overshoot
  runs past both ends of the range, so the saturating clamp is exercised at 0
  and at 255 rather than assumed.
- **The DOWNSCALE arm specifically**, because the support scaling only exists
  there, and because a chat request almost always downscales.

**Reachability** is the served case, not the unit gate. A chat request whose
image is NOT a multiple of the fixture's `factor` returns 400 before W6c and 200
with the placeholder count the resized grid implies after it, through
`ApiServer::handle_chat_completions` on the server's default configuration.

**And the served regime has to be the DOWNSCALE one, which it was not at first.**
`Dots3NoteResizedSize` only rounds each side to a multiple of `factor`, so every
resize the 6x14 fixture produces is an UPSCALE on both axes: `filterscale` is 1,
the support stays 2.0, and the resampler degenerates to exactly the textbook
four-tap cubic §4.13.1 spends its length arguing it is not. Measured: 6x14 ->
8x16 is BYTE-IDENTICAL with the support scaling deleted. The one downscale that
reached `ProcessImage`, 9x9 -> 8x8, carries a CONSTANT image, which every
normalized weight set preserves by construction. So M3 — dropping the single
property this unit exists for — left every `ProcessImage` and every served
assertion passing, and only the isolated unit cases caught it. Since `factor` is
28 on the released checkpoint, essentially every real request downscales, and
the regime proven through the production entry point was the one users never
hit.

The repair is the fixture's `p_max_pixels`, lowered to `kBudgetMaxPixels = 64`,
which is how production reaches this regime too: the budget comes off
`preprocessor_config.json`. 24x96 under it resolves to 4x16 — a 6x downscale on
both axes and a 25-tap support-scaled window — over a TEXTURED image, in two
places. `test_dots3_note_vision`'s "ProcessImage DOWNSCALES through the
support-scaled window" is the numeric arm against the independent reference, and
it is what M3 has to RED. `test_openai_api_server_dots3_mm_forward`'s case 2e
serves it over HTTP and carries the same twin-equality reachability arm as 2d,
with the same limit: both legs run the same resampler, so a defect inside it
still cancels there.

**A local Pillow cross-check is recorded as evidence and is NOT the gate.**
Pillow is not in the oracle table and carries no pin here, so a comparison
against the copy installed on this host cannot gate anything. It is still the
strongest single piece of evidence available that the port reads the algorithm
correctly — it is the very library upstream calls — so the measurement is
recorded with its exact version and treated as evidence, not as a result.

#### 4.13.4 Evidence

Measured 2026-09-03 on the developer host, CPU build,
`-DVLLM_CPP_SERVER=ON -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF
-DCMAKE_BUILD_TYPE=Release`.

**Every number in this section was RE-MEASURED on the repaired tree**, the one
that carries the budget-downscale arms, at branch head `3fb874179` plus this
change. The earlier table (base `origin/main` `e24805924`, vision
`2f2b3798a663dcf9…` 13/13 21242, server `52ad05417f0d4a28…` 15/15 216) described
the tree before those arms existed and is superseded rather than merged, because
a table that mixes two trees cannot be read.

**The clean tree.**

| target | binary sha256 | cases | assertions |
|---|---|---|---|
| `test_dots3_note_vision` | `f99fefb17e85f3b8…` | 13 / 13 | 21343 / 21343 |
| `test_openai_api_server_dots3_mm_forward` | `17797ed0144e8134…` | 16 / 16 | 240 / 240 |

**The two reference arms.** Per-pixel MAXIMUM, in levels of 0-255, over 18 cases
(six geometries by three images: a hard 0/255 checker, a wrapping ramp, and
high-frequency noise):

| arm | measured | bound |
|---|---|---|
| `impl` vs the FIXED-POINT reference | **0** levels | 0 (exact) |
| `impl` vs the CONTINUOUS reference | **1.07588** levels | 2.0 |

The continuous bound of 2 is derived rather than fitted: the horizontal pass
rounds by at most half a level, the vertical pass carries that through weights
whose absolute sum is at most **1.268771** (the swept supremum, §4.13.1), and
then rounds again, so the ceiling is `0.5 * 1.268771 + 0.5 = 1.13439` and
1.07588 sits under it. The asserted gate bound stays 2.0, which remains safe.
This derivation first read 1.125 from the wrong `sum |k|` of 1.25; the corrected
ceiling is larger and both the measured value and the asserted bound survive it.

`ProcessImage` against the same reference, reconstructed back through the
per-channel normalization and `pre_pixel_shuffle`'s transpose: **0** levels on
the served 6x14 image (an upscale to 8x16) and **0** levels on the 24x96 image
the `kBudgetMaxPixels` budget downscales 6x to 4x16.

**Reference independence.** Every qualified name in the CODE inside `namespace
ref_resample` (170 lines) enumerated: **2 distinct, 45 occurrences, both
`std::`** — `std::size_t` (27) and `std::vector` (18). Zero non-`std::` names,
which is the same test W6a's and W6b's references passed. One further
`ref_resample::` occurs in a COMMENT, the one that says `ResizeContinuous` and
`ResizeExact` share `Weights`; it is prose about the arms, not a call.

**RED-before at the production entry point.** On the commit that landed the
resampler without calling it, `test_openai_api_server_dots3_mm_forward` read
`14 | 12 passed | 2 failed`, `192 | 190 passed | 2 failed`, both failures
`REQUIRE(r.status == 200)` against the processor's own HTTP 500:

```text
what=Dots3NoteImageProcessor: image requires resize (14x6 -> 16x8); the bicubic
resize path is not ported (W6a uses conformant images). Owed by W8 and tracked
by issue #2537
```

**Mutations.** Each applied to the clean tree, rebuilt, run, and the tree
restored to zero modified files. The binary sha is recorded beside the case
counts because a changed sha alone does not prove the mutation reached the code.

| # | mutation | vision sha256 | vision | server sha256 | server |
|---|---|---|---|---|---|
| — | *(clean)* | `f99fefb17e85f3b8…` | 13/13, 21343/21343 | `17797ed0144e8134…` | 16/16, 240/240 |
| M1 | delete the `PilResizeBicubicRgb` call site | `a667d66950aa5737…` | **12/13, 2 failed** | `64d218c55e4f47d2…` | **14/16, 8 failed** (cases 2d and 2e) |
| M2 | half-pixel centre (`(xx + 0.5) * scale` -> `xx * scale`) | `a02a243c3d18944e…` | **12/13, 182 failed** | `c951890b3289f116…` | 16/16, 240/240 |
| M3 | drop the support scaling (`filterscale = 1.0`) | `8b69d5db92b71d70…` | **12/13, 121 failed** | `acdc3b15a4cdbfe3…` | 16/16, 240/240 |
| M4 | skip the per-output weight normalization | `e1cecc05459e1927…` | **11/13, 214 failed** | `d6e4f531a41ba72e…` | 16/16, 240/240 |

**Per-arm, which is where M3's story actually is.** The `ProcessImage`
reconstruction reports its worst level for each geometry, so the two arms can be
read separately:

| mutation | 6x14 -> 8x16 (upscale) | 24x96 -> 4x16 (6x downscale) |
|---|---|---|
| *(clean)* | 0 | 0 |
| M1 | **255** | **143** |
| M2 | **148** | **31** |
| M3 | **0 — BLIND** | **119** |
| M4 | **30** | **154** |

M3 is the reason the downscale arm was added. Before it, the upscale arm was the
ONLY `ProcessImage` case, and it reads 0 under M3: on an upscale
`filterscale = max(1, in/out)` is 1, so deleting the support scaling deletes
nothing, and 6x14 -> 8x16 comes out byte-identical. The served suite reads 0
failures under M3 as well. Dropping the single property this unit exists for was
therefore invisible everywhere except the isolated `PilResizeBicubicRgb` unit
cases, on a model whose `factor` is 28 and whose every real request downscales.

**M1 found a real hole and the repair is part of this brick.** On the first
pass the served suite stayed at `14 | 14 passed`, `199 | 199 passed` with the
call site disabled, which is exactly the failure
[reachability.md](../reachability.md) names: both served cases were blind to it
by construction, one asserting geometry that `Dots3NoteResizedSize` already
owned and one asserting only that two different images differ. Case 2d serves
the 6x14 image and then serves the 8x16 buffer the resampler produces from it —
conformant, so it takes the identity path — and requires the two logprob vectors
to be equal element for element. With it, M1 fails the served suite. Case 2e is
the same arm on the budget-downscaled geometry, and M1 fails both.

**How M1 fails in case 2d, precisely, because it is not a value difference.**
With the resize call deleted the 2d leg patchifies an 8x16 grid out of the
252-byte 6x14 buffer at `rowstride = rw * 3 = 48`, so it reads up to byte 383 of
a 252-byte allocation — 132 bytes past the end. Case 2d therefore detects M1
through a HEAP OVER-READ whose bytes happen to differ, not through a defined
value difference, and under a sanitizer the mutant would abort rather than fail
an assertion. Production cannot reach that state: the only caller that can skip
the resize is the size-equality short circuit, where the two strides agree. It
is recorded because "the mutation reddened the gate" and "the mutation reddened
the gate for the reason the gate claims" are different statements.

**Case 2e does not have that shape, which is a second reason it is worth its
cost.** Its M1 leg patchifies a 4x16 grid out of the 6912-byte 24x96 buffer, so
the largest byte it reads is 191 — entirely in bounds. The 4x16 window it reads
is the top-left corner of the source instead of the resample of the whole image,
so 2e detects M1 through a DEFINED value difference. Under a sanitizer 2e still
fails as an assertion where 2d aborts.

**M2, M3 and M4 stay GREEN on the served suite, and that is correct rather than
a gap.** Both legs of cases 2d and 2e run the same resampler, so a defect inside
it cancels. The served suite answers "was it called"; the reference gate answers
"was it right". The test file says so in the same words, because a green served
suite under three of the four mutations reads like coverage it does not have.
What case 2e adds over 2d is not a numeric verdict but the REGIME: without it no
served request, and no `ProcessImage` call on a textured image, ever reached the
support-scaled window at all.

**The local Pillow cross-check — EVIDENCE, not a gate.** Pillow is not in the
oracle table and carries no pin here, so nothing measured against it can gate.
It is still the library upstream itself calls, so the port was compared to it
directly: a standalone harness linking `pil_resize.cpp` against
`PIL.Image.Image.resize(..., Image.Resampling.BICUBIC)` from the Pillow
**12.1.1** installed on this host, over **436 cases** — 36 hand-picked
geometries by four image kinds, plus a 400-case fuzz over random extents in
`[1, 120]` on both axes with random content. **436 of 436 agreed byte for byte;
worst absolute difference 0 of 255.** Read that as "this port reads Resample.c
correctly on this Pillow", never as a parity result: a different Pillow could in
principle move, and the committed gate is the in-tree reference above.

**THE HARNESS IS NOT RETAINED, so do not go looking for it in the tree.**
Committing it would add a Pillow build dependency to a repository that has none,
for a comparison that is not allowed to gate, so it was run and discarded. The
consequence is stated rather than hidden: this number is not reproducible from a
checkout, and anyone who wants it again rebuilds the harness from §4.13.1's step
list. A fresh review of this row re-ran the same comparison independently over
490 cases — the 18 committed geometries, 18 edge geometries including 1-pixel
sides and 64x64 -> 1x1, and 400 random extents — and read 490 of 490 byte-exact,
worst absolute difference 0.

**What was NOT measured.** No GPU, and none is relevant — the resample is host
CPU work upstream of the tower. No throughput number on any axis. And no
comparison to vLLM, which §6.2 records as impossible on any host this project
owns.

---

### 4.14 W7a puts the AUDIO tower on a SERVED request, and refuses the rest by name

**Issue: [#2703](https://github.com/mudler/vllm.cpp/issues/2703). Brick: W7a,
the first slice of W7.** After this slice an OpenAI `input_audio` chat part
travels the whole production chain — `ApiServer::handle_chat_completions` ->
the dots3-note chat seam -> the mel front end -> `AsyncLLM` -> the scheduler's
encoder admission -> `EncodeMmDots3NoteForCausalLM` -> the 32-layer `dots`
speech encoder -> `EmbedMmDots3NoteForCausalLM`'s masked scatter -> the
language forward — and the answer depends on the waveform. Before it, the same
request died at the entrypoint with HTTP 400 "At most 0 audio(s) may be provided
in one prompt.", because `Dots3NoteChatSupportedMmLimits()` declared only
`{"image", 1}`.

**#2703 IS THE OWNING ISSUE AND IT IS NOT VERBATIM CORRECT.** A scoping pass
between filing and implementation corrected it in four places, and this section
is the record of that rather than a silent divergence:

| #2703 says | Measured | Where |
|---|---|---|
| `audio_encoder.py:507-519` for the absent positional embedding | `:498-510` | §2.5, re-measured at `9035151d6` |
| `nvidia/audio_encoder.py` is 736 lines and the anchors in §2.5 stand | eight §2.5 anchors were `+9` | §2.5 |
| the adapter is "`audio_adapter.proj.{0, 1, 3}`, 1280 to 5120" | `proj.0` is a **LayerNorm with weight AND bias** over 1280, and only `proj.1`/`proj.3` are Linears | the committed shard index |
| "It must be ported from whatever upstream actually calls" | it is **already ported**: `audio_processor.cpp:211-319` is Whisper's `log_mel_spectrogram` verbatim in double precision, and `test_voxtral_e2e.cpp:157-178` already drives it at n_fft 400 / hop 160 / n_mels 128 / 16 kHz | §4.14.2 |

#### 4.14.1 What was already built, and what was actually missing

The audio chain was mostly present before this slice. Measured, unchanged by it,
and named here so a reviewer does not look for edits that are not in the diff:
the runner's `execute_mm_encoder` is modality-blind; the scheduler's encoder
admission is modality-blind; `EmbedMmDots3NoteForCausalLM` is modality-blind and
its scatter balances because the audio adapter's `whisper_adapter_out_dim` is
5120, which IS `config.hidden_size`; `MultiModalFeatureSpec::audio_data`
(`inputs.h:81`) already carries an `AudioKwargs`; `DecodeInputAudioPart`
(`chat_mm.cpp:122-129`) already base64-decodes the part; `DecodeWavPcm16Mono`
(`audio_processor.cpp:103-131`, over the shared chunk walk `ParseWavPcm16` at
`:49-99` that W7c-1 factored out of its body) already decodes PCM16 mono WAV;
and
`ExpandAudioPlaceholders` (`audio_processor.cpp:326-345`) already performs the
expansion.

Five things were missing or dead, and they are what W7a adds:

1. `mm_chat_dots3note.cpp`'s supported-limit map declared `{"image", 1}` only,
   so an `input_audio` request was refused at `ValidateChatMmLimits`.
2. There was no dots3 audio processor: no slaney bank at [201, 128], no
   `pad_or_trim` to 960000, no `ceil(samples / 1280)` token count, no marker-id
   resolution.
3. There was no audio tower.
4. The loader deferred all 430 `audio_encoder.*` tensors
   (`dots3_note.cpp:603-604`).
5. `EncodeMmDots3NoteForCausalLM` refused `modality != "image"`
   (`dots3_note_registry.cpp:188-193`).

**`RouteAudioWav` (`chat_mm.cpp:131-160`) is left ALONE**, deliberately. It is
DEAD in `src/` — nothing outside `tests/` calls it — but its test is the
`ROAD-V1-MM` parse gate, and it hard-codes Whisper's fixed
`max_source_positions` token count, which is the WRONG rule for dots3. Editing
it would move another row's gate to serve this one. W7a writes
`RouteDots3NoteAudioWav` beside `RouteDots3NoteImageRgb` instead, in the
architecture's own TU, which is the same shape W6a chose over editing
`RouteImageRgb`.

#### 4.14.2 The mel front end was already ported, and it is gated against a REAL oracle

`log_mel_spectrogram` (`audio.py:117-126` @ `9035151d6`) is Whisper's, verbatim:
periodic Hann over `n_fft` 400, `torch.stft(center=True)` reflect padding, the
last frame dropped by `stft[..., :-1]`, a POWER spectrogram, `filters @
magnitudes`, `clamp(1e-10).log10()`, a GLOBAL-max `-8` floor and `(x + 4) / 4`.
`WhisperAudioProcessor::ProcessWaveform`
(`src/vllm/multimodal/audio_processor.cpp:211-319`) is that function, in double
precision, and the only deltas dots3 needs are CONFIG: `chunk_length_s` 30 ->
60, `n_mels` 80 -> 128, `max_source_positions` 1500 -> 6000. So W7a REUSES it
rather than writing a second one, which is what "never write a parallel path by
hand" asks for. 960000 samples / hop 160 = 6001 STFT frames, minus the dropped
last one = 6000 = `chunk_seconds * 100`, which is the assert upstream itself
makes at `audio.py:215`.

**NO FFT IS ADDED.** Both in-tree front ends compute a direct DFT of the 201
needed bins per frame and record that as a deviation
(`audio_processor.cpp:8-11`, `parakeet_audio_processor.h:23-27`); the difference
from `torch.stft` is float summation order. W7a claims no performance axis, so
importing an FFT would be work no gate on this row could read.

**THE FILTERBANK IS PROMOTED TO A SHARED SEAM, AND IT GETS THE ONE REAL ORACLE
THIS ROW HAS.** The tree carried SIX mel implementations and no shared audio
front-end seam. `ParakeetMelFilterBank`
(`parakeet_audio_processor.cpp:65-106`) is already a faithful double-precision
`mel_filter_bank(norm="slaney", mel_scale="slaney", min 0, max sr/2)` and is
parameterised rather than Parakeet-specific, so W7a extracts it as
`vllm::multimodal::MelFilterBankSlaney` in a new
`include/vllm/multimodal/mel_filter_bank.h` and has BOTH callers use it. The
Whisper/dots3 orientation is `[num_frequency_bins, num_mel_filters]` — upstream
`mel_filter_bank`'s own — and `MelFilterBankSlaneyTransposed` returns Parakeet's
`[num_mel_filters, num_frequency_bins]`. Transposition reorders `float`s and
does not round, so Parakeet's two existing gated tolerances are byte-identical
by construction rather than by tolerance.

The oracle: `tests/vllm/multimodal/fixtures/voxtral_audio/
voxtral_mel_filters_f32.bin` is COMMITTED, is exactly 102912 bytes = 201 x 128
`float32`, and was dumped by `scripts/mm/a3_voxtral_oracle_capture.py:141-147`
from `mistral_common.audio.mel_filter_bank(num_frequency_bins=201,
num_mel_bins=128, min_frequency=0.0, max_frequency=8000.0,
sampling_rate=16000)`. That is the same call `audio.py:98-106` makes for dots3.
**Measured before a line of this slice was written: the double-precision
construction reproduces all 25728 values BIT-FOR-BIT — max ULP difference 0,
max absolute difference 0.0.** That settles HTK-versus-Slaney, the `norm`
argument and the integer-divided-Nyquist detail (`np.linspace(0, sampling_rate
// 2, num_frequency_bins)`) outright, and it is the only place on this row where
a number is checked against something a third party produced rather than against
a reference this repository also wrote.

#### 4.14.3 The tower is a NEW FILE, mirroring `nvidia/audio_encoder.py`

`src/vllm/model_executor/models/dots3_note_audio.{h,cpp}`, beside
`dots3_note_vision.{h,cpp}` and for the same reason: upstream itself forks
`modeling_whisper.py` into a separate file rather than parameterising Whisper.
`WhisperAudioEncoderForward`
(`include/vllm/model_executor/models/whisper_audio.h`) is READ and NOT extended
— it carries the identical pre-norm block skeleton and the exact
q/v/out-bias-and-NOT-k convention, and W7a routes through the same `vt` ops it
does — but the deltas are structural, not parametric: RMSNorm instead of
LayerNorm, a packed-SwiGLU MLP instead of GELU, partial RoPE instead of a fixed
additive sinusoid, and a 3-layer Conv2d stem instead of two Conv1ds.

**The 430 tensors, verified against the committed shard index rather than taken
from prose.** All BF16; no FP8 anywhere, so W9 has nothing to say about this
tower.

| Count | Shape | Name |
|---|---|---|
| 1 | `[480, 1, 3, 3]` + `[480]` | `conv2d1.weight` / `.bias` |
| 2 | `[480, 480, 3, 3]` + `[480]` | `conv2d2`, `conv2d3` |
| 1 | `[1280, 7680]` | `conv_out.weight`, **no bias** |
| 32 | `[1280]` | `layers.{L}.self_attn_layer_norm.weight` (RMSNorm, no bias) |
| 32 | `[1280, 1280]` + `[1280]` | `layers.{L}.self_attn.q_proj` |
| 32 | `[1280, 1280]` | `layers.{L}.self_attn.k_proj`, **no bias** |
| 32 | `[1280, 1280]` + `[1280]` | `layers.{L}.self_attn.v_proj` |
| 32 | `[1280, 1280]` + `[1280]` | `layers.{L}.self_attn.out_proj` |
| 32 | `[1280]` | `layers.{L}.final_layer_norm.weight` |
| 32 | `[10240, 1280]` + `[10240]` | `layers.{L}.fc1` — the PACKED SwiGLU pair |
| 32 | `[1280, 5120]` + `[1280]` | `layers.{L}.fc2` |
| 1 | `[1280]` | `layer_norm.weight` |
| 1 | `[1280]` w + b | `audio_adapter.proj.0` — a **LayerNorm** |
| 1 | `[5120, 1280]` + `[5120]` | `audio_adapter.proj.1` |
| 1 | `[5120, 5120]` + `[5120]` | `audio_adapter.proj.3` |

**No learned positional embedding**, as §2.5 records. Position enters as PARTIAL
RoPE: `rotary_dim = int(head_dim * partial_rotary_factor)` rounded down to even
= 32 out of head_dim 64, theta 10000, NeoX half-split
(`audio_encoder.py:55-79`, `140-166`). `vt::RopeFromCache` supports exactly this
through `RopeArgs::rotary_dim`, with the `[P, rotary_dim]` cos|sin cache it
already consumes for the vision tower.

**THE CONV2D STEM IS COMPOSED AS im2col + `vt::MatmulBT`, AND THAT IS ONE EXACT
TRACKED EXCEPTION TO THE `kConv2d` SEAM.** `vt::Conv2d` exists, and it has
exactly ONE provider: `src/vt/cpu/cpu_conv2d.cpp:111` is the sole
`RegisterOp(OpId::kConv2d, ...)` in the tree. Routing the stem through it would
build a capability that faults the moment anyone runs this model on a CUDA
queue — which is every host that could serve it. `whisper_audio.h:33` already
composes its Conv1d stem this way and says "no new CUDA kernel", so this is
in-tree precedent rather than an invention. The missing CUDA provider is a real
`vt` gap that outlives this row and it has its own issue, cited in the code
beside the exception.

**THE TEMPORAL MASK IS NOT OPTIONAL.** `_conv2d_stem_one_chunk`
(`audio_encoder.py:535-562`) zeroes the padded tail at FOUR stages — before
`conv2d1`, and after each of the three GELUs — with
`valid_mel_lens = audio_sample_lens // 160` halved by `(n + 1) // 2` at each
stride-2 layer (`:570-574`). It is load-bearing and a shape check cannot see it:
the mel of a ZERO-padded tail is not zero, it is the `-8` floor pushed through
`(x + 4) / 4`, a nonzero constant, and without the mask that constant leaks
through the 3x3 receptive fields into the LAST VALID tokens. The chain also
lands exactly: 16000 samples -> 100 valid mel frames -> 50 -> 25 -> 13, and
`ceil(16000 / 1280)` is 13.

#### 4.14.4 `MlpGateUpMethodBase` gets a BIAS arm, and that is the seam rule applied

The released audio MLP is `fc1 [10240, 1280]` **with a `[10240]` bias**, then
`swiglu` (`audio_encoder.py:42-44`: `x1, x2 = x.chunk(2, -1); silu(x1) * x2`,
which is gate-then-up and therefore `vt::SiluAndMul`'s own order), then
`fc2 [1280, 5120]` with a `[1280]` bias. `layers::MlpGateUpMethodBase`'s three
existing members all return `silu(gate) * up` from WEIGHTS ALONE, so the seam as
it stood could not express this MLP at all. AGENTS.md says to extend a shared
seam exactly when it cannot represent the upstream behaviour, so W7a adds
`layers::UnquantizedMlpGateUpBiasMethod`: the merged `[2I, H]` operand plus a
`[2I]` bias, one `vt::MatmulBT`, one row-broadcast `vt::Add`
(`ops.h:3486-3495`), one `vt::SiluAndMul`.

**Every existing caller is byte-identical BY CONSTRUCTION, not by tolerance.**
The three existing methods are not touched; the bias arm is a fourth derived
class on the same base, which is precisely how
`UnquantizedMlpGateUpSplitMethod` and `UnquantizedMlpGateUpGeluMethod` were
added. The vision suites are re-run at their current counts to prove the
no-bias path did not move.

**THIS DOES NOT CLOSE [#2616](https://github.com/mudler/vllm.cpp/issues/2616),
and saying so is the honest record.** #2616 is titled "dots3-note VISION tower:
`use_bias=true` still refuses" and its own closing condition is three things:
the seam arm, loading the vision `qkv`/`proj`/`fc` biases, and deleting the
vision refusal, gated by a `use_bias` fixture served end to end. W7a lands the
FIRST of the three, and it lands it REACHED — the audio checkpoint is what
reaches it, which is exactly the "it would land unreached" objection #2616
raises against doing the seam work on its own. The other two are still owed to
the vision arm and no published checkpoint sets `use_bias`. Writing `Closes
#2616` on this pull request would leave a record saying the vision tower accepts
a config it still refuses by name.

#### 4.14.5 What W7a refuses, BY NAME, and to which brick

| Refused | Named brick | Where |
|---|---|---|
| ~~audio longer than `chunk_seconds` (60 s = 960000 samples)~~ | **LIFTED by W7b (§4.15)** | was `Dots3NoteAudioProcessor`; the segment loop replaced it, and what stands in its place is the §4.15.3 geometry refusal |
| any container but PCM16 mono RIFF/WAVE | **W7c** | the route, before decode |
| ~~any sampling rate but `audio_config.sampling_rate` (16000)~~ | **LIFTED by W7c-2 (§4.17)** | was `Dots3NoteAudioProcessor::ProcessWaveform`; `ResampleAudioScipy` stands in its place, and what is refused there now is upstream's own `pyav` arm |
| `use_causal == true` | unshipped arm | `ParseDots3NoteAudioParams`, at INSTALL |
| `use_conv1d_stem` (`use_conv2d_stem == false`) | unshipped arm | same |
| `use_latent_input == true` | unshipped arm | same |
| `merge_factor != 1` | unshipped arm | same |
| `encoder_type != "dots"` | upstream refuses it too (`audio.py:255-256`) | same |
| `use_rms_norm == false` | unshipped arm | same |
| `use_rope == false` | unshipped arm (it would need `embed_positions`) | same |
| a marker id that the tokenizer does not carry | none — unservable | the chat seam, at INSTALL |
| `whisper_adapter_out_dim != config.hidden_size` | none — unservable | same |
| more than ONE audio part | this seam's own ceiling | `ValidateChatMmLimits` |

**THE CONFIG REFUSALS ARE AT INSTALL, NOT IN THE ENCODER, and that is a
measurement rather than a taste.** `mm_chat_dots3note.cpp:232-240` records what
happened when this row threw from inside `encode_mm`: it runs in the engine's
busy loop, stopping `AsyncLLM` and turning every LATER request — TEXT ONES
INCLUDED — into a 500. `Dots3NoteAudioRefusalFor(config)` is the install-time
twin of `Dots3NoteVisionRefusalFor`, and the encoder keeps its own check as
defence in depth on the same polarity. **The refusal predicate and the route
predicate are THE SAME PREDICATE**, called from both places, because a refusal
that is narrower than the route it guards is a silently wrong answer rather than
an error.

**Three upstream knobs are DEAD, not deferred, and the difference is worth
writing down.** `conv_chunksize`, `conv_bucket_max_elements` and
`conv_bucket_step` are read out of the config by `Dots3NoteAudioConfig`
(`audio.py:47`, `:51-52`) and copied onto the `WhisperConfig`
(`audio.py:160-165`), and then nothing in `audio_encoder.py` reads any of the
three: `_forward_conv2d_stem` calls `_conv2d_stem_one_chunk` once, on the whole
tensor. They are set-and-never-read at `9035151d6`. Recording them as DEAD says
a later brick owes nothing; recording them as deferred would invent a debt.

**`chunk_seconds` is the one refusal whose absence would be SILENTLY WRONG
rather than merely incomplete.** Upstream's tower chunks a long waveform into
60-second segments and sums `ceil(chunk_len / 1280)` PER SEGMENT
(`audio.py:141-146`), while the prompt side computes one `ceil(total / 1280)`
(`processor.py:771`). For a waveform at or under one chunk the two agree
exactly, and W7a's `ceil(samples / 1280)` is therefore upstream's own number.
Past one chunk they diverge, the placeholder span stops matching the tower's row
count, and a masked scatter that does not balance splices audio features onto
text rows. W7b owns the segmentation; until it lands, the refusal is what keeps
the two sides equal.

**W7b HAS LANDED, and §4.15.3 records what replaced this refusal.** The segment
loop makes the tower's row count `sum_i ceil(seg_i / 1280)`, and that sum equals
the prompt side's one `ceil(total / 1280)` exactly when `chunk_samples` is a
whole number of `token_stride`s — which the released config satisfies and the
tiny fixture's own `chunk_seconds = 1` does not. A waveform spanning more than
one chunk on a config that does not satisfy it is now what refuses by name.

#### 4.14.6 The three marker ids come from the TOKENIZER, and the checkpoint was checked

`processor.py:757-760` resolves `audio_start_id` / `audio_pad_id` /
`audio_end_id` out of the TOKENIZER'S VOCAB by string, and
`multimodal.py:82-89` reads `added_tokens.json` off the checkpoint directly. The
strings are `audio_config`'s `audio_comp_start` / `audio_comp_span` /
`audio_comp_end`, defaulting to `<|audio_comp_start|>` / `<|audio_comp_pad|>` /
`<|audio_comp_end|>` (`audio.py:37-39`). W7a mirrors that: it resolves the three
by string against the installed tokenizer's added tokens and REFUSES BY NAME
when one does not resolve, rather than defaulting to an id.

**The released checkpoint was checked, and it carries them.**
`dots-studio/dots3-note-prev` at revision
`1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b`, file `added_tokens.json`
(sha256 `1aa71a4e0dbab80a72fd925389fd6c9cc52d1cb9da5dee8282784c15c6fa789b`, 2795
bytes, 85 entries), has `<|audio_comp_start|>` = **151718**,
`<|audio_comp_end|>` = **151719**, `<|audio_comp_pad|>` = **151720**. Note the
ORDER: start, end, pad. A port that assumed the three were consecutive in
start/pad/end order would have produced a syntactically valid prompt with the
pad and end ids swapped, and no shape check anywhere could see it. Resolving by
string is what makes that impossible rather than merely unlikely.

The marker the chat seam injects is `get_placeholder_str`'s audio branch
(`multimodal.py:68-69`): `<|audio_comp_start|><|audio_comp_pad|><|audio_comp_end|>`,
with the SINGLE pad in the middle expanded to N by `ExpandAudioPlaceholders`.

#### 4.14.7 The gate is a CONSISTENCY gate, and it has TWO independent references

§6.4 option B applies unchanged: this row has no oracle and will not get one, so
correctness is argued by in-test double-precision references written from the
upstream Python and sharing NO helper with the implementation. The row's
convention is to PROVE that independence by enumerating every qualified name in
the reference namespace (W6a 70, W6b 105, W6c 45 — all `std::`).

**W7a splits the reference in two rather than writing one.** A single reference
covering the DFT, the mel bank and a 32-layer tower with a four-stage temporal
mask is too large for a reviewer to hold, and the two halves fail in unrelated
ways: the front end's hazards are windowing, framing and normalisation, the
tower's are ordering, masking and bias placement. So there is a FRONT-END
reference and a TOWER reference, each with its own enumerated name list, and
both counts are reported.

The one place a real oracle exists — the filterbank against
`voxtral_mel_filters_f32.bin` — is asserted to float32 rounding, and §4.14.2
records that it came back bit-exact.

#### 4.14.8 Reachability

Production entry point: `ApiServer::handle_chat_completions` on the server's
DEFAULT configuration. The smallest failing test enters through it with an
`input_audio` part and goes RED at head with HTTP 400 "At most 0 audio(s) may be
provided in one prompt."

**The load-bearing case is the two-different-waveforms LOGPROB one**, for the
reason `test_api_server_dots3_mm_forward.cpp:33-38` already records for images:
status 200, `prompt_tokens` and `completion_tokens` ALL PASS on a tree where the
tower is replaced by a correctly SHAPED constant. The logprobs of the first
generated token do not.

The mutations, each RED first, each restored byte-for-byte, each reported with
the rebuilt binary's sha256 AND the doctest case counts — a changed sha alone
proves a rebuild, not that the mutation reached the code:

| # | Mutation | What it proves |
|---|---|---|
| A | delete the tower call in `encode_mm` | the encoder hook reaches the tower |
| B | tower -> correctly-shaped constant | the gate reads VALUES, not shapes |
| C | delete the temporal mask | §4.14.3's leak is measured, not argued |
| D | give `k_proj` a bias | the q/v/out-and-NOT-k asymmetry is asserted |
| E | delete the production loader materialisation call site | the weights come from the loader, not from the test |

#### 4.14.9 Risks

- **The temporal mask's four stages are easy to get to three.** Mutation C
  deletes them; the gate reports the resulting error so the reader can see how
  large the leak is rather than only that it exists.
- **`fc1` is a PACKED pair and the halves are not interchangeable.** Gate-then-up
  is `vt::SiluAndMul`'s order and `x.chunk(2, -1)`'s order, and swapping them
  produces a correctly-shaped wrong answer. The tower reference computes the
  swap explicitly and asserts the two differ.
- **Partial RoPE rotates HALF of each head.** A port that rotated all 64 dims
  would still produce [T, 1280]. The reference rotates 32 and asserts the
  untouched tail is bit-equal to the input's tail.
- **The mel front end is shared with Whisper/Voxtral.** Any change to
  `WhisperAudioProcessor` now moves two models. The Voxtral suite is re-run and
  its counts reported.

#### 4.14.10 Stop conditions

Stop and report `NEEDS_DECISION` if the bias arm cannot be added without
changing an existing caller's behaviour, or if the released tokenizer turns out
not to carry the three markers. Stop and report `NEEDS_CONTEXT` rather than
guessing a value the config does not carry.

#### 4.14.11 Evidence, measured on the merge commit

Measured on the W7a branch after `origin/main` was merged in at `8853af6bf`,
because a merge can falsify a claim made before it. Host: 20-core x86-64, CPU
arm only (`-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SERVER=ON -DVLLM_CPP_BUILD_TESTS=ON
-DCMAKE_BUILD_TYPE=Release`). Every number below was produced by the session
that reports it; none is inherited.

**The suites, by real target name.**

| Target | Cases | Assertions |
|---|---|---|
| `test_dots3_note_audio` | 13 / 13 passed | 1980 / 1980 |
| `test_openai_api_server_dots3_mm_forward` | 23 / 23 passed | 287 / 287 |
| `test_dots3_note_vision` | 13 / 13 passed | 21343 / 21343 |
| `test_dots3_note_scaffold` | 26 / 26 passed | 110835 / 110835 |
| `test_dots3_note_attn` | 51 / 51 passed | 6888 / 6888 |
| `test_parakeet_audio_processor` | 6 / 6 passed | 41054 / 41054 |
| `test_parakeet_encoder` | 7 / 7 passed | 543 / 543 |
| `test_parakeet_ctc_engine` | 2 / 2 passed | 12485 / 12485 |
| `test_parakeet_transcription_fold` | 4 / 4 passed | 38 / 38 |
| `test_parakeet_transducer` | 3 / 3 passed | 777 / 777 |

`test_whisper_audio`, `test_gemma4_vision_tower` and `test_gemma4_audio_tower`
each report **1 case and 0 assertions**, and `test_voxtral_e2e` exits **77**.
Those are the tree's own environment-gated skips — they want
`VLLM_WHISPER_ENC_WEIGHTS` and `VLLM_VOXTRAL_SAFETENSORS`, and the voxtral one
prints "GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass". They are recorded
as NOT RUN, not as green. §4.14.9's risk paragraph asked for the Voxtral suite's
counts as the shared-front-end proof and this host cannot produce them: the
shared bank's executable evidence is the bit-exact oracle case below, plus
Parakeet's 41054.

**The one real oracle.** `MelFilterBankSlaney(201, 128, 0.0, 8000.0, 16000)`
against the committed `voxtral_mel_filters_f32.bin`: **0 of 25728 values differ,
worst |delta| 0** — bit for bit. Measured beside it, and the reason "25728 agree"
is not 25728 independent facts: the bank is **sparse, 394 of 25728 nonzero**.

**D2 did not move Parakeet.** No Parakeet TEST file is in this change
(`git diff origin/main...HEAD --stat` names only
`src/vllm/multimodal/parakeet_audio_processor.cpp` on that side), so the 41054
assertions are the same population as on `main` and they pass. The extraction
moved every arithmetic line unchanged and rounds once, on the same
`static_cast<float>`; `MelFilterBankSlaneyTransposed` only reorders floats
already rounded.

**D5 left every existing caller byte-identical, structurally.**
`git diff origin/main...HEAD -- include/vllm/model_executor/layers/linear.h`
is **68 insertions and 0 deletions**: `UnquantizedMlpGateUpBiasMethod` is a
FOURTH derived class and not one existing line of the seam changed, so "no bias"
is not a new branch. `linear.h` is the only file the seam change touches.

**Both reference-independence counts, MEASURED.** The enumeration is computed
from the test file's own bytes with comments and string literals stripped:

| Reference | Distinct | Occurrences | Scopes |
|---|---|---|---|
| `ref_front` | 11 | 71 | `std` only |
| `ref_tower` | 6 | 53 | `std` only |

Neither reference reaches `vllm::`, `vt::` or the file under test. The counts
the file first carried — 22 and 19 — were **wrong**, and the case that asserted
them could not tell: it compared two hand-written constants with two literals.
Eleven of `ref_front`'s listed names are unused in that namespace and two
(`std::llround`, `std::int16_t`) appear nowhere in the file except in the list
naming them. Repaired in the same change; the property the counts supported was
true throughout.

**The agreement measurements.** Front end vs `ref_front`: worst |delta|
**7.22919e-08**. Tower vs `ref_tower`: rel-L2 **0.00770442**, output spanning
**[-1.80469, 2.15625]** rather than a constant. The mask stages measured
**50 -> 25 -> 13 -> 7** against a padded mel of 100 frames and a stem output of
13, and the padded tail sits at **-0.660975, not at 0** — which is why §4.14.3's
leak is real. At 1281 samples the span is **2** and the mask **1**, so the two
numbers are not derivable from one another.

**The five mutations.** Each applied to a clean tree (the source blob's hash is
compared before and after, because a mutation that never applied reads as a
passing test), rebuilt, run, then restored and re-verified by blob hash. A
build failure was treated as a build failure and never as a red: mutation A's
first form died on `-Werror=unused-variable` and was corrected before it could
be counted. Binary sha256 is reported **with** case counts, because a changed
sha proves a rebuild and not that the mutation reached the code.

Baseline (both suites GREEN):
`test_dots3_note_audio` sha256 `6f0eebce3aeca1de…`, 13/13, 1976/1976;
`test_openai_api_server_dots3_mm_forward` sha256 `d3de4496830eac5c…`, 23/23,
287/287.

| # | Mutation | Audio suite | Served suite | Binary sha256 (audio / served) |
|---|---|---|---|---|
| A | delete the tower call in `encode_mm` | 13/13, 1976/1976 — GREEN | **22 passed, 1 failed**; 286/1 | `9ace2f3f51eff5fb…` / `bc7ad5e37e2fb621…` |
| B | tower -> correctly-shaped constant | **8 passed, 5 failed**; 1970/6 | **22 passed, 1 failed**; 286/1 | `4d448c702b43aa66…` / `07728ad2a25d96f0…` |
| C | delete the four temporal-mask stages | **10 passed, 3 failed**; 1972/4 | 23/23 — GREEN | `5be72bd7f6275098…` / `156b403295bebf33…` |
| D | give `k_proj` q's bias | **12 passed, 1 failed**; 1975/1 | 23/23 — GREEN | `c84ce66473c1279d…` / `3dcbd601e1f0b79f…` |
| E | delete the loader materialisation call site | 13/13, 1976/1976 — GREEN | **20 passed, 3 failed**; 275/4 | `455d91b72ab0078d…` / `dc76f6e09b8c5803…` |

Read the GREEN cells, because they are the point. **A reddens only the SERVED
suite, and inside it only the two-waveforms LOGPROB case** (`CHECK(worst >
1e-4)`, `test_api_server_dots3_mm_forward.cpp:1313`): status, `prompt_tokens`
and `completion_tokens` all still pass with the tower call deleted, exactly as
§4.14.8 predicted. **E reddens only the SERVED suite**, on
`REQUIRE(r.status == 200)` — a tower-only gate cannot see a deleted production
call site, which is what "Nothing lands dead" asks. **C and D redden only the
TOWER suite**: the served logprob case compares two waveforms, and two waveforms
still differ when the mask is gone, so the served gate is honestly blind to
them. No single suite detects all five.

After restoration both binaries rebuilt to the **byte-identical baseline
sha256** (`6f0eebce3aeca1de…`, `d3de4496830eac5c…`) and `git status` is clean.

**The tokenizer markers were verified against the released checkpoint, not
assumed.** `dots-studio/dots3-note-prev` `added_tokens.json` (sha256
`1aa71a4e0dbab80a72fd925389fd6c9cc52d1cb9da5dee8282784c15c6fa789b`) and
`tokenizer.json` (sha256
`7f4e21a1d9fa472439f70201b4849977da5ec11e73df5a36552ab5ee99af554b`, 85 added
tokens) both carry all three as SPECIAL added tokens:
`<|audio_comp_start|>` **151718**, `<|audio_comp_end|>` **151719**,
`<|audio_comp_pad|>` **151720**. Note the order: `pad == start + 2`, and
`pad != start + 1`. A port that assumed start/pad/end consecutive would build a
well-formed wrong prompt. The code resolves all three BY STRING from the
tokenizer and refuses BY NAME when one does not resolve.

**The released checkpoint's audio tensors, re-measured from the committed
index.** 430 tensors under `audio_encoder.`, **all BF16 and not one F32**;
`k_proj.bias` **absent** while `q_proj.bias`, `v_proj.bias` and `out_proj.bias`
are present 32 times each; `fc1.bias [10240]` and `fc2.bias [1280]` 32 times
each, which is the caller that makes D5's arm REACHED; `conv_out.bias` absent.

**Upstream anchors re-read at `9035151d6`**, in `~/_git/vllm` at
`vllm/models/dots3_note/nvidia/audio_encoder.py`: the file is **736 lines**
(§2.5's `+9` correction holds); `fc1`/`fc2` at `:334-335` take torch's default
`bias=True`; the conv2d stem is `:466-474`; `self.embed_positions = None` is
`:508`; the four mask stages are `:544-561` and
`valid_mel_lens = audio_sample_lens // hop_length` is `:570-574`;
`_temporal_mask` at `:528-533` keeps `arange(T) < valid_lens`, which is what
`MaskTime` zeroes from `valid` onward; and `k_proj` alone is `bias=False` at
`:221`.

#### 4.14.12 What the FRESH REVIEW measured, and the one hole it found

The review returned PASS on the tower's arithmetic, on its refusals and on the
oracle comparison, and it mutated each of them rather than reading them. What
follows is what it came back with that the sections above did not already say.
Only the first item is a defect. The rest are not, and they are written down
anyway, because a number nobody records is a number the next reader has to
derive again.

**THE ENUMERATION INSTRUMENT HAD A `'` HOLE, AND IT IS NOW CLOSED.**
`StripCommentsAndLiterals` treated EVERY `'` as a char-literal delimiter, so a
C++14 digit separator opened a literal that was never there and the scan ran on
to the next `'`, dropping the real code in between. Two separators bracketing a
`vt::` call therefore hid that call from the enumeration, and §4.14.7's
independence property read GREEN while being false. The stripper now takes a
pp-number WHOLE. The number must START at a digit that does not continue an
identifier, so `u8'a'` and `L'x'` are still char literals and are still
stripped — which is why the fix is the pp-number rule and not the shorter
"ignore a `'` after an alphanumeric": that shorter rule would leak the body of
every prefixed char literal instead.

Measured on this host, on ONE unfixed binary and ONE fixed binary, each row a
source-text edit against an UNREBUILT binary. The injected reach is the same
`vt::Scale(vllm::kMelFloor)` in all three reaching rows; only what brackets it
changes.

| # | Injected into `ref_front`, one line, at the same point | Unfixed `49cd8fb95403aff3…` | Fixed `8856bcf28d6e9070…` |
|---|---|---|---|
| M-A1 | the reach, bare | **RED**, 3 axes: `std,vllm,vt != std`, `13 != 11`, `73 != 71` | **RED**, the same 3 axes |
| M-A2 | the same tokens inside a `//` COMMENT | GREEN, 11 / 71, `std` | GREEN, 11 / 71, `std` |
| M-A3 | ONE separator: `16'000.0 * <reach>` | RED by COUNTS only, `8 != 11` and `50 != 71`; the scope set wrongly read `std` | **RED**, all 3 axes |
| M-A4 | the reach BRACKETED: `16'000.0 * <reach> / 1'280.0` | **GREEN**, 11 / 71, `scopes=std`, with a LIVE `vt::` call inside the namespace — the hole | **RED**, all 3 axes |

M-A3 loses names rather than gaining them because the phantom literal its lone
`'` opens finds no closing `'` before the end of `ref_front`, so the scan drops
the whole tail of the namespace. That is the same defect as M-A4 and it happens
to be loud. M-A4 is the quiet form, and the quiet form is the one that matters:
adding a real `vt::` reach moved NOTHING.

Clean source on both binaries: `ref_front` 11 distinct / 71 occurrences,
`ref_tower` 6 / 53, `std` only. The fix moves neither count.

Every row above is a SOURCE-TEXT edit re-run against an UNREBUILT binary, which
is the fact `DOTS3_AUDIO_TEST_SOURCE` (`tests/CMakeLists.txt:1268`) exists to
make true: the instrument reads bytes at run time, so an edit reaches it without
a compile, and the unchanged binary sha256 is what proves the reading rather
than a compiled-in transcription. The hole was NOT live at the head that found
it — the file carries no digit separator outside a comment, and M-A1 shows a
naive reach is caught — but a later edit that wrote `16'000` and `1'280` into a
reference with a helper call between them would have reopened it in silence, and
on this row that property IS the correctness argument, because there is no
oracle. The intermediate binary carrying the fix WITHOUT the standing
assertions below, `8081852cbd8aecd4…`, produced the same four rows.

**AND THE FIX HAS A STANDING GATE, not only a mutation.** A source mutation
proves the hole once; it does not stop the next rewrite of the stripper from
reopening it, and nothing on a clean tree can, because the whole point of the
repair is that the counts do NOT move. So the enumeration case now calls
`StripCommentsAndLiterals` directly on three strings, in four assertions: the
bracketed reach
`16'000.0 * vt::Scale(vllm::kOne) / 1'280.0`, whose two qualified names must
SURVIVE; a prefixed `u8'v'`, whose body must NOT; and a `//` comment carrying
both a `vt::` token and a separator, which must go whole. RED FIRST, with the
assertions in place and the pp-number rule reverted: binary
`f440d09b4dee9919…`, that case **0 passed / 1 failed, 18 of 20 assertions**,
both `find` CHECKs red and the prefix and comment CHECKs green. GREEN after,
binary `8856bcf28d6e9070…`: **13/13, 1980/1980**. The source was restored
byte-for-byte between the two, and the restored tree rebuilt to the same
`8856bcf28d6e9070…`. Those four assertions are why the suite reads 1980 at this
head where §4.14.11's table records the 1976 it measured at ITS tree.

**THE BUILD-FAILURE TRAP, IN ITS SHARPEST CONCRETE FORM.** §4.14.11 already
records that mutation A's first form died on `-Werror`. The review produced the
same failure with the consequence visible: its first form of mutation C died on
`-Werror=unused-function`, and because the build failed, the binary still on
disk was the one the previous mutation had left there — **byte-identical, sha256
`87182e77c28535df…`** — and it reported **13/13 GREEN**. A reader who checked
only the doctest line would have recorded "mutation C is not caught". Checking
the ninja return code AND the binary sha256 is what separates NOT CAUGHT from
NEVER BUILT, and this row has now been bitten by that distinction twice.

**MEASURED HEADROOM ON THE TOWER BOUND, RECORDED AND NOT ACTED ON.** The tower
case asserts `rel < 5e-2` (`test_dots3_note_audio.cpp:1132`) against a measured
**0.00770442**, so the bound sits about **6.5x above the baseline**. A defect
that moves the answer by less than that survives it, and a PARTIAL bias defect —
bias applied to only the `up` half of the packed `fc1` pair, say — is plausibly
inside that band. It is recorded as headroom rather than tightened, because
tightening it without measuring what a tighter bound costs in false reds on a
bf16 envelope would trade one unmeasured risk for another. The bound's own
justification is unchanged: a 2-block tower and a 3-layer conv stem in bf16,
plus the deliberate `vt::RmsNorm` rounding difference `dots3_note_audio.h`
records.

**ONE CAVEAT THE INSTRUMENT DOES NOT COVER, STATED PLAINLY.** `ref_tower` is fed
`LoadedTower::mel_ref` (`test_dots3_note_audio.cpp:1083-1088`), which is a
double-promoted copy of the IMPLEMENTATION'S mel and not a second computation of
it. That is legitimate layering — the front end is separately gated against
`ref_front` to **7.22919e-08**, so the tower case is deliberately measuring the
tower and not the front end twice — but the enumeration proves a property of the
two references' CODE and says nothing about the tower reference's INPUT. "Two
independent references" is therefore true of what they compute, and not of what
they are handed.

**A SIXTH MUTATION, FROM THE REVIEWER, WORTH KEEPING BESIDE THE FIVE.** Deleting
the `vt::Add` in `UnquantizedMlpGateUpBiasMethod::Apply` (`linear.h:215`) — the
D5 arm still SELECTED and still named `bf16-gate-up-bias`, but behaving as
no-bias — is CAUGHT: rel-L2 **0.0077 -> 0.0987** against the `5e-2` bound. It is
a FOURTH tower-only defect the served suite cannot see, alongside C and D, and
it is the mutation that proves the new seam arm's bias is applied rather than
merely reachable.



### 4.15 W7b lifts the `chunk_seconds` refusal, so a real recording is served

**Issue: [#2797](https://github.com/mudler/vllm.cpp/issues/2797). Brick: W7b,
on top of W7a ([#2703](https://github.com/mudler/vllm.cpp/issues/2703), merge
`e5efa29f0`).** W7a decodes a WAV `input_audio` part, makes 128 mel bins, runs
the 32-layer `dots` speech encoder and scatters the result into the prompt
embeddings — for a clip at or under `chunk_seconds`, 60 s on the released
config, and refuses anything longer BY NAME (§4.14.5). W7b writes the segment
loop that refusal names. It is what makes the audio path usable on a recording
rather than on a clip.

#### 4.15.1 Upstream, and the exact `file:line@SHA` ported

`dots3_note` does not exist at the parity pin `5559679229`, so every anchor
below names `9035151d6`, read in the local clone `~/_git/vllm`
(`git rev-parse 9035151d6` = `9035151d6c9fb726181469f9e6aa9ccbf9a5dacb`).

| Upstream | What it is | Ported to |
|---|---|---|
| `nvidia/audio.py:193-234` | `DotsEncoderWithMask.encode_waveform` — the whole brick | `Dots3NoteAudioProcessor::SegmentWaveform` + `ProcessWaveform` + `Dots3NoteAudioForwardChunks` |
| `nvidia/audio.py:196-203` | the `while time_step * SAMPLE_RATE < n` slicing loop | `SegmentWaveform` |
| `nvidia/audio.py:209-212` | the per-segment `token_len` | `SegmentWaveform`, through the UNCHANGED `NumAudioTokens` |
| `nvidia/audio.py:213-218` | `pad_or_trim` + `log_mel_spectrogram` + the `chunk_mel_frames` assert, per segment | `ProcessWaveform`'s loop body |
| `nvidia/audio.py:220-227` | `torch.stack` of the mels, `input_seq_lens = token_lens * merge_factor`, ONE encoder call | `Dots3NoteAudioForwardChunks` |
| `nvidia/audio.py:229-234` | keep `[idx, : token_len * merge_factor, :]` of each chunk, `torch.cat` | `Dots3NoteAudioForwardChunks`'s concatenation |
| `nvidia/audio.py:129-147` | `compute_audio_token_length` — upstream's own chunked TOTAL | the identity §4.15.3 gates; DEAD upstream (`git grep` at `9035151d6` finds one hit, its own `def`) |
| `nvidia/audio_encoder.py:664-685` | the varlen PACK: `cu_seqlens`, the valid-token mask, the gathered rope positions | §4.15.2 — why the loop IS this |
| `nvidia/audio_encoder.py:711-719` | the varlen UNPACK back to `[B, max_seqlen, D]` | same |
| `nvidia/audio_encoder.py:570-577` | `valid_mel_lens = audio_sample_lens // hop_length`, PER BATCH ELEMENT | the per-chunk `num_samples` W7a's tower already takes |
| `common/processor.py:762-771` | the PROMPT side's one `ceil(total / stride)` | the UNCHANGED `NumAudioTokens`, and §4.15.3's invariant |

#### 4.15.2 A loop over W7a's tower IS upstream's batched varlen call

Upstream batches the chunk mels and makes ONE encoder call. This port calls
W7a's single-chunk `Dots3NoteAudioForward` once per chunk and concatenates. That
is the same function, not an approximation, and the reason is the varlen path
itself:

1. **The stem never mixes chunks.** `_forward_conv2d_stem` takes
   `[B, 1, n_mels, T]` and masks each batch element from ITS OWN
   `valid_mel_lens` (`audio_encoder.py:570-577`, `:545-561`). Conv2d is
   batch-independent.
2. **Attention never crosses a chunk.** The pack builds `cu_seqlens_q` from
   `input_seq_lens.cumsum` (`:674-677`) and hands it to
   `flash_attn_varlen_func` (`:276`), so each chunk is its own bidirectional
   window. That is `AttentionDenseFlash` over one chunk, which is what W7a
   already runs.
3. **The rope positions RESTART at 0 per chunk.** `position_ids` is
   `arange(S)` over the PADDED stem length (`:644-646`), and the pack then
   gathers `token_positions.expand(B, S)[valid_token_mask]` (`:679-685`) — the
   first `token_len` positions of EACH row. So chunk 7's first token carries
   position 0, not `7 * 750`. W7a's cache is built at `arange(num_tokens)` and
   is therefore already the right cache for every chunk.
4. **Everything after the layers is ROW-WISE.** The unpack writes the kept rows
   back into a zero-filled `[B, max_seqlen, D]` (`:711-719`); `layer_norm`
   (`:721`), the adapter's LayerNorm, its two Linears and its GELU
   (`audio.py:240-248`) all act on one row at a time, and the zero rows are
   sliced away by `[idx, : token_len, :]` (`:229-234`) before anything reads
   them.

So the ONLY thing the batch buys upstream is one kernel launch instead of `k`.
This port pays `k` launches and computes the same numbers. **W7a's single-chunk
path is byte-identical under this change** — at `k == 1` the loop calls the same
function with the same arguments — which is what makes W7b additive rather than
a rewrite, and it is also the answer to the NEEDS_DECISION the dispatch named:
the pack/unpack CAN be mirrored without touching that path, because for one
chunk it degenerates to a prefix slice and for many it degenerates to `k`
independent prefix slices.

**The log-mel is computed PER SEGMENT and that is not an optimisation.**
`log_mel_spectrogram` floors at `log_spec.max() - 8.0` (`audio.py:124`), a
GLOBAL max over the tensor it is given. Upstream gives it one padded segment at
a time (`:213-214`), so the floor is per-chunk. A port that ran the front end
once over the whole waveform and sliced the mel afterwards would use ONE max for
every chunk, and the difference is a per-chunk additive shift on the quietest
bands of the quietest chunk. This port calls the same
`WhisperAudioProcessor::ProcessWaveform` per segment, which is upstream's shape.

#### 4.15.3 The one arithmetic invariant, and the new refusal that guards it

`NumAudioTokens` IS NOT TOUCHED, and #2797 records why. Upstream's per-segment
count is `(segment_length - 1) // (HOP_LENGTH * conv_temporal_stride *
merge_factor) + 1` (`audio.py:210-212`) and W7a wrote `ceil(n / 1280)`. These
are algebraically identical for every `n >= 1`. They differ at `n == 0`, and
only in C++: Python's `//` floors, so `(0-1)//1280 + 1 == 0`, while C++ integer
division truncates toward zero, so a literal transcription yields
`(-1)/1280 + 1 == 1` — one phantom token for an empty segment. W7b ports the
identity, not the characters. The reference in the gate carries upstream's
LITERAL expression, and it is safe there for a reason worth writing down: the
slicing loop's `while time_step * SAMPLE_RATE < n` condition (`:196`) means no
segment is ever empty, so the `n == 0` case the two forms disagree on is
unreachable from either side.

The invariant W7b must hold is between TWO DIFFERENT upstream expressions:

* the PROMPT side counts `math.ceil(total / stride)` in one go
  (`processor.py:771`), which is what `NumAudioTokens(total)` computes and what
  the placeholder span is built from;
* the TOWER produces `sum_i ceil(seg_i / stride)` rows, which is upstream's own
  `compute_audio_token_length` (`audio.py:129-147`).

Write `C = chunk_samples`, `s = token_stride` and `n = total`. Every segment but
the last is exactly `C` long, so the sum is `k * ceil(C/s) + ceil(rem/s)` and
the prompt side is `ceil((kC + rem)/s)`. **The two are equal for every `n`
exactly when `C % s == 0`**, and they differ otherwise: at `C = 16000`, `s =
1280` (the tiny fixture's own geometry) a 2.5-chunk waveform gives 13+13+7 = 33
rows against a span of `ceil(40000/1280) = 32`.

The released config satisfies it — `chunk_samples` 960000 = 750 * 1280 — and so
does every EVEN `chunk_seconds` at 16 kHz, because `16000 * cs % 1280 == 0` iff
`cs` is even. A config that does not satisfy it is one upstream itself would
splice on, since upstream runs both expressions and never compares them. This
port compares them and REFUSES BY NAME, per request, when the waveform actually
spans more than one chunk:

> the waveform needs `k` chunks and this checkpoint's `audio_config` gives
> `chunk_samples` that is not a whole number of `token_stride`s, so the
> placeholder span and the tower's row count would differ.

**It is a per-request refusal and not an install-time one, deliberately.** The
predicate is a property of the REQUEST as much as of the config: at `chunk_samples
% token_stride != 0` a single-chunk clip is still served correctly, because a
one-segment sum is `ceil(n/s)` on both sides. Refusing the whole audio capability
at install would refuse clips upstream serves. It is raised from
`ProcessWaveform`, which runs in the CHAT SEAM before the engine — the same place
W7a's rate refusal is raised, as an `InputValidationError` mapping to HTTP 400 —
and NOT from `encode_mm`, where a throw runs in the engine's busy loop and turns
every later request, text ones included, into a 500 (§4.14.5).

#### 4.15.4 What changes, file by file

| File | Change |
|---|---|
| `include/vllm/multimodal/inputs.h` | `AudioKwargs` gains `num_chunks` (default 1) and the two per-chunk length vectors. `input_features` becomes `[num_chunks, n_mels, n_frames]`; at `num_chunks == 1` the layout is byte-identical to W7a's and every pre-W7b producer and consumer is unchanged |
| `include/vllm/multimodal/dots3_note_processor.h` | `Dots3NoteAudioProcessor::AudioChunk` and `SegmentWaveform`, the production seam a gate can drive without a front end; `ProcessWaveform`'s contract |
| `src/vllm/multimodal/dots3_note_processor.cpp` | `SegmentWaveform` (`audio.py:196-212`); `ProcessWaveform` loops (`:208-218`); the `chunk_seconds` refusal is REPLACED by the §4.15.3 one |
| `src/vllm/model_executor/models/dots3_note_audio.{h,cpp}` | `Dots3NoteAudioForwardChunks` (`audio.py:220-234`), which is the production entry point from W7b on. `Dots3NoteAudioForward` keeps its arithmetic and gains ONE check: upstream's `assert mel.shape[1] == self.chunk_mel_frames` (`audio.py:215`), made executable there, with `chunk_seconds` carried on `Dots3NoteAudioParams` as `DotsEncoderWithMask` carries it (`audio.py:169-171`). §4.15.6 records why: without it M5 is GREEN |
| `src/vllm/model_executor/models/dots3_note_registry.cpp` | `EncodeAudioDots3Note` calls the chunked function |
| `src/vllm/entrypoints/openai/mm_chat_dots3note.cpp` | the comment over `ProcessWaveform` names the refusal that is left, and records that this call is the FRONT END and not the engine loop |
| `docs/FEATURES.md`, `docs/USAGE.md` | both stated the `chunk_seconds` ceiling as owed to W7b; both now state what is served and what the divisibility invariant refuses |
| `tests/vllm/models/dots3_note_tiny_fixture.h` | a length-parameterised long clip, and the `a_chunk_seconds` knob it drives |
| `tests/vllm/models/test_dots3_note_audio.cpp` | `ref_chunks`, the third reference namespace, and the seam cases |
| `tests/vllm/entrypoints/openai/test_api_server_dots3_mm_forward.cpp` | the served multi-chunk request and its two-waveform LOGPROB case |

Every OTHER refusal is kept and stays owed: non-PCM16-mono-WAV and non-16 kHz
(W7c), `use_causal`, `use_conv1d_stem`, `use_latent_input`, `merge_factor != 1`,
`encoder_type != "dots"`, `use_rms_norm == false`, `use_rope == false`.

#### 4.15.5 Gates — chunking needs shapes single-chunk did not

No oracle (§6.4 option B), so correctness rests on the in-test double-precision
references. W7a has two; W7b adds a THIRD, `ref_chunks`, for the segmentation
GEOMETRY — the offsets, the per-segment lengths, the per-segment token counts
and the row offset of each chunk in the concatenation. It is written from
`audio.py:196-234` and uses nothing but `std::`. The heavy numerics are NOT
re-written: the driver hands each segment to `ref_front::LogMel` and to the
existing `RefTower`, both unchanged, at the geometry `ref_chunks` derived.
**W7a's enumeration instrument is EXTENDED to cover it** — the same
`QualifiedNamesIn` reading this file's own bytes at run time, with the scope-set
assertion and the two counts, now over three namespaces — rather than a second
instrument being written.

**A tolerance on the concatenated output cannot see any of these, and each
produces correctly-shaped output**, so each gets a SEAM assertion rather than an
aggregate one:

| Defect | What the aggregate sees | The seam assertion |
|---|---|---|
| off-by-one in the per-chunk `token_len` slice | a norm that is still small | per-chunk row COUNTS from `SegmentWaveform` against `ref_chunks`, and the concatenated row at each boundary against that chunk's reference row 0 |
| chunks concatenated in the WRONG ORDER | nothing — every row is a correct row | each chunk's block of the output compared against ITS OWN reference block, and the deliberately reversed concatenation asserted to DIFFER |
| the short final chunk padded but not truncated | a longer output, but every row well-formed | the last chunk's row count asserted `< ` the full chunks', and the total asserted equal to the placeholder span |
| the temporal mask taken from the PADDED length | a small norm change | the last chunk driven at both lengths in the reference and asserted to DIFFER on its LAST KEPT row |

**The geometry is chosen so none of the four can alias.** `a_chunk_seconds = 2`
(32000 samples, 25 token strides — §4.15.3's condition holds), a waveform of
80000 samples = 5 s = **2.5 chunks**: THREE chunks, of which the last is
genuinely SHORT, with per-chunk token counts **25, 25, 13** summing to 63 =
`ceil(80000/1280)`. Three chunks means a reversal is not a swap of two equal
halves; a short last chunk means the truncation is exercised; and 80000 is not a
multiple of `chunk_samples`, which is what makes the last chunk short.

#### 4.15.6 Reachability

Production entry point `ApiServer::handle_chat_completions` on the default
configuration, with an `input_audio` part longer than `chunk_seconds`. That
request is HTTP 400 at W7a's head — "SEGMENTATION IS NOT PORTED" — which is the
RED. The two-different-waveforms LOGPROB case is mandatory and is the
load-bearing one: status and token counts pass on a tree whose tower is a
correctly-shaped constant.

What the served suite CANNOT do here is recorded by W7a and is not re-learned:
it gates REACHABILITY, not tower arithmetic — four separate tower-only defects
leave it green (§4.14.12). Chunk-seam defects are gated at the
processor/tower level against the references; the served case proves reach.

The mutations, each RED-first, each restored byte-for-byte, each reported with
the binary's sha256 AND the case counts:

| # | Mutation | Where |
|---|---|---|
| M1 | reverse the chunk order in the concatenation | `Dots3NoteAudioForwardChunks` |
| M2 | drop the final short chunk's truncation (keep all stem rows) | same |
| M3 | off-by-one the per-chunk slice | same |
| M4 | compute the temporal mask from the PADDED length | `ProcessWaveform`'s per-chunk `num_samples` |
| M5 | delete the production call that lifts the refusal | `EncodeAudioDots3Note` |

**M5 MEASURED A REACHABILITY HOLE AND CLOSED IT, and that is the finding of this
brick.** Written as a pure deletion M5 is trivial. Written as the substitution it
has to be — put W7a's `Dots3NoteAudioForward(mel.input_features, mel.num_samples,
mel.num_tokens, ...)` back, so nothing in production reaches
`Dots3NoteAudioForwardChunks` — the served suite stayed GREEN and the tower suite
stayed green with it. The reason is that the flattened call produces a
correctly-SHAPED answer with the RIGHT ROW COUNT: it reads the stacked mel as one
600-frame mel, takes 75 stem rows and returns the first 63, which is exactly the
placeholder span. Two different waveforms still give two different logprobs, so
even the load-bearing case cannot see it. A gate that stays green without the
call site measures a class, not a capability, so the hole was closed rather than
recorded: `Dots3NoteAudioForward` now carries upstream's own
`assert mel.shape[1] == self.chunk_mel_frames` (`audio.py:215`), which is the one
number that separates one chunk from a stack of them, and M5 is RED at the
entrypoint. A case drives that check directly so it is not a mute switch.

#### 4.15.7 Risks

1. **The divisibility invariant is silent when it fails.** Mitigated by
   §4.15.3's refusal and by a case that drives the tiny fixture's OWN
   non-divisible geometry and asserts the refusal names it.
2. **`AudioKwargs` is shared with Whisper's `RouteAudioWav`.** Mitigated by
   defaulting `num_chunks` to 1 and leaving the two vectors empty, so the
   pre-W7b layout is the `num_chunks == 1` case and no other producer changes.
3. **A per-chunk encoder call costs `k` launches where upstream pays one.** No
   performance axis is claimed on this row (§0), and the alternative is a
   batched tower this port has no other caller for.

#### 4.15.8 Stop conditions

Stop and report `NEEDS_DECISION` if the varlen pack/unpack cannot be mirrored
without changing W7a's single-chunk path (it can — §4.15.2), or if the geometry
in the issue, this spec and the fixture disagree.

#### 4.15.9 Evidence, measured 2026-09-03

Host: the developer's x86-64 Linux box, CPU queue, `-DVLLM_CPP_SERVER=ON
-DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`. No GPU
lease was taken and no number here is a performance number. The build is SCOPED
to the two targets this brick moves, because the box was at 97% on `/` and a
full-tree link on this row has already produced ENOSPC failures that read like
real defects; the build tree lives in `/dev/shm`.

Oracle identity asserted before any citation: `git rev-parse HEAD` in
`~/_git/vllm` is `5559679229bc961848b121ccdeaa8fa5d79bec98`, this project's
parity pin, which carries no `dots3_note`; every anchor below therefore names
`9035151d6c` (`[Model] Add native Dots3 NOTE multimodal support (#51255)`), read
out of that same checkout with `git show`.

| Suite (REAL target name) | Result |
|---|---|
| `test_dots3_note_audio` | **20 cases, 3869 assertions, 0 failed** (19 / 3858 before the reachability repair) |
| `test_openai_api_server_dots3_mm_forward` | **26 cases, 311 assertions, 0 failed** |

Baseline binaries: `test_dots3_note_audio`
`bad4cfd1c4ac035c3e73a8db217364dc5b06b8f5f3080b651d789c9070bcd280`,
`test_openai_api_server_dots3_mm_forward`
`0e5ac1a9201da4f2a7314ac195290423e52f940104fced6e4db46398386e0c0f`. Both were
recomputed after the last mutation was restored and match byte for byte.

**RED at W7a's production head.** The seven production files were checked out at
`e64f00560` — the merge base — with the tests, the fixture and the docs kept, and
`test_openai_api_server_dots3_mm_forward` was rebuilt
(`54ab3d22f4aba33312fb14fe7466b67f062422d164d24a0812b504f9be0e8b80`). It failed
**3 cases / 7 assertions**, the two reach cases on `REQUIRE(r.status == 200)` and
the refusal case on the message, with the body carrying "SEGMENTATION IS NOT
PORTED and is owed to W7b". The tower suite cannot be built at that head, because
`SegmentWaveform`, `AudioChunk` and `Dots3NoteAudioForwardChunks` do not exist
there and a build failure is not a red.

**The independence instrument, extended rather than duplicated**, reading this
file's own bytes at run time: `ref_front` 11 distinct / 71 occurrences,
`ref_tower` 6 / 53, `ref_chunks` 2 / 25, every scope `std` and nothing else.

**The mutations**, each applied by a harness that asserts the mutation APPLIED
and the build SUCCEEDED before it reads a result, each restored byte-for-byte:

| # | `test_dots3_note_audio` | `test_openai_api_server_dots3_mm_forward` | audio binary sha256 |
|---|---|---|---|
| M1 reverse the chunk order | **1 case / 9 assertions FAILED** | 26 / 311 pass | `f149ecef2a1915ff663000d206ad0181fa6d3603b4b2a116d308e0dbb9361d06` |
| M2 no truncation of the short chunk | **2 cases / 2 assertions FAILED** | **2 cases / 2 FAILED** (500: "75 embedding rows for a placeholder span of 63") | `8d84f47495ac4d0df41eaf03112d498ea3daf2bd78862911a78d89fa284d6715` |
| M3 off-by-one the per-chunk slice | **2 cases / 12 assertions FAILED** | 26 / 311 pass | `44f2cd8c49aa278e8c249efb57f40b5c24615ff4a889fa206bdd35b9cc9940f5` |
| M4 mask from the PADDED length | **3 cases / 14 assertions FAILED** | 26 / 311 pass | `7bdc11e55ed400b13babd04b8a57f1c776b252b25a04ccbb2c8f589960286ae3` |
| M5 delete the production call site | 20 / 3869 pass | **2 cases / 2 FAILED** (500: "the mel holds 9600 values, which is not ONE chunk of 16 x 200") | `87d9ae650bb38818a9906ae8dd205e310962f612d102bd48035c7d199ed77ff5` |

Every sha differs from the baseline and from every other row, so no result is a
stale binary reporting green. The M1/M3/M4 column of served greens is not a gap:
it is §4.14.12's measurement reproduced — the served suite gates REACHABILITY and
not tower arithmetic — and it is why the seams are gated where they are.

**The geometry gated**: `chunk_seconds` 2 = 32000 samples = 200 mel frames = 25
token strides; a clip of 80000 samples = 5 s = 2.5 chunks -> **32000, 32000,
16000** samples contributing **25, 25, 13** rows = 63 = `ceil(80000/1280)`. The
boundary walk drives 1, 1280, 31999, 32000, 32001, 80000 and 96000 samples and
asserts, at each, that the segments TILE the waveform and that the per-segment
sum equals `NumAudioTokens`. The refusal case drives the fixture's OWN
non-divisible `chunk_seconds` 1 at 40000 samples, where the sum is 33 against a
span of 32, and asserts that a clip inside one chunk is still served there.


### 4.16 W7c-1 accepts a MULTI-CHANNEL WAV at 16 kHz, by upstream's own mean

**Issue: [#2813](https://github.com/mudler/vllm.cpp/issues/2813). Brick: W7c-1,
the CHANNELS half of W7c.** W7c was one brick over two unrelated questions. The
channel arm is an exact mirror of a two-line upstream reduction with no oracle
risk; the rate arm needs a resampler and a recorded divergence. Keeping them
apart stops one claim borrowing the other's credibility, so W7c-1 lifts the
channel refusal ALONE and W7c-2 keeps the rate one.

Before this slice, a PCM16 WAV **already at `audio_config.sampling_rate`** was
refused with HTTP 400 for one reason: its `fmt ` chunk said two channels and
`DecodeWavPcm16Mono` threw "not mono". Nothing about the model, the front end or
the tower was missing. After it, the same file is served, and the waveform the
tower sees is the per-sample mean over its channels.

#### 4.16.1 Upstream, and the exact `file:line@SHA` ported

`dots3_note` does not exist at the parity pin `5559679229`, so every anchor
below names `9035151d6`, read in the local clone `~/_git/vllm`
(`git rev-parse 9035151d6` = `9035151d6c9fb726181469f9e6aa9ccbf9a5dacb`;
`git rev-parse HEAD` = `5559679229bc961848b121ccdeaa8fa5d79bec98`).

| Upstream | What it is | Ported to |
|---|---|---|
| `vllm/multimodal/media/audio.py:207-208` | `if mono and y.ndim > 1: y = np.mean(y, axis=tuple(range(y.ndim - 1)))` — the DECODE-side reduction, over the CHANNEL axes of a `(channels, samples)` array | `DecodeWavPcm16MeanToMono`'s inner reduction |
| `vllm/multimodal/media/audio.py:220` | `load_audio(..., mono: bool = True)` — the default that makes `:207-208` reached, not optional | why the reduction is unconditional here |
| `vllm/multimodal/media/audio.py:168-169` | the SAME `np.mean(audio, axis=0)` on the PyAV fallback arm, so both decoders agree | evidence the mean is the format-independent rule |
| `vllm/multimodal/audio.py:46-52` | `ChannelReduction`, whose `MEAN` member carries the comment "(default, preserves energy balance)" | the choice of mean over `FIRST`/`MAX`/`SUM` |
| `vllm/multimodal/audio.py:69-70` | `AudioSpec.target_channels: int | None = 1`, `channel_reduction: ChannelReduction = ChannelReduction.MEAN` | the defaults |
| `vllm/multimodal/audio.py:150-152` | `normalize_audio`: `if spec.target_channels == 1: if spec.channel_reduction == ChannelReduction.MEAN: ...` | the PARSER-side reduction, which selects the same operation |
| `vllm/multimodal/parse.py:697-700` | `MultiModalDataParser` applying `normalize_audio(new_audio, AudioSpec(target_channels=self.target_channels))` | the call site `:150-152` is reached from |
| `vllm/models/dots3_note/common/processor.py:523-525` | `get_data_parser` returning `MultiModalDataParser(target_sr=..., target_channels=1)` | **dots3-note's own selection of that spec** |

**One anchor in #2813 was wrong and is corrected here rather than silently.**
The issue writes the last row as `common/processor.py:523-525`, which reads as
`vllm/multimodal/common/processor.py`. No such file exists at `9035151d6`
(`git ls-tree -r --name-only 9035151d6 -- vllm/multimodal/` lists 26 paths and
none is `common/`). The file is
`vllm/models/dots3_note/common/processor.py`, and its `:523-525` is exact. The
same short form is what two production strings in this tree carried, so a reader
chasing the anchor landed nowhere; §4.16.4 repairs those.

#### 4.16.2 The arithmetic, and the type it is done in

Upstream reduces a `float32` array: `soundfile.read(dtype="float32")` makes
each `int16` sample exactly `s / 32768`, and `np.mean` over a `float32` input
accumulates in `float32`. This port does not have a float32 array to reduce —
it has the interleaved `int16` frames of the `data` chunk — so the intermediate
type is a decision that has to be stated.

**It accumulates the raw `int16` channel samples in `int32`, divides once in
`double`, and narrows once to `float`.**

**The two claims below use different domains, and each one names its own.**
The overflow claim is about the PARSER'S domain, `C <= 65535`, because
`channels` is a `uint16` field of the `fmt ` chunk. The bit-identity claim is
about `C <= 512`. An earlier draft of this section stated both in the same
paragraph without saying so, which read as one claim over 65535 channels and was
false above 512.

- The `int32` accumulator **cannot overflow anywhere in the parser's domain**.
  `|s| <= 32768` and `C <= 65535`, so `|acc| <= 32768 * 65535 = 2147450880
  < 2^31`. The sum is therefore EXACT for every WAV this parser can be handed,
  which no `float32` accumulator can promise past **512** channels.
- **The answer is the correctly-rounded `float` of the exact mean, for every
  `C` in that domain — but there are TWO roundings, not one.** `acc` and
  `32768 * C` are both exact in `double`, so the divide is correctly rounded to
  `double`; the narrowing store then rounds a second time whenever that quotient
  is not itself exact, which is every `C` that is not a power of two. Saying
  "exactly one rounding" was imprecise about the MECHANISM. The CONSEQUENCE
  survives, for two independent reasons: `double`'s 53 bits clear the
  `2 * 24 + 2 = 50` that makes a division's double rounding innocuous, and an
  exhaustive sweep of **all 1,300,542,267 `(C, acc)` pairs** for every
  non-power-of-two `C` in `[2, 200]` — every accumulator value those channel
  counts admit — finds **zero** anomalies against an 80-bit route. So the
  shipped expression is the correctly-rounded `float`, and nothing is rounded to
  `float` and then combined again.
- **For a power-of-two channel count UP TO 512 that second rounding does not
  happen either, and the result is BIT-IDENTICAL to what upstream's `float32`
  mean produces, in whatever order `numpy` sums.** With `C = 2^k` the divisor
  `32768 * C` is `2^(15+k)` and `|acc| <= 2^(15+k)`, so the quotient is
  `acc * 2^-(15+k)`; every integer of magnitude at most `2^24` is exact in
  `float32`, so this is exact exactly when `15 + k <= 24`, that is
  **`k <= 9`, `C <= 512`**. Upstream's arm is exact under the same bound: each
  `s / 32768` is exact, and every partial sum — in ANY grouping, so pairwise
  summation included — is a multiple of `2^-15` whose numerator is bounded by
  the same `2^24`, after which the divide by `2^k` only moves the exponent.
  This covers **C = 1**, which is why no mono waveform in this tree can move,
  and **C = 2**, which is the case this slice exists to serve, and every channel
  count a WAV container plausibly carries.
- **The 512 bound is TIGHT, and it is the measured one rather than the
  derivation's.** The derivation as first written — "a significand of at most
  `16 + k` bits against `float`'s 24" — gives `k <= 8`, `C <= 256`. It is
  conservative by one step: the only value that would need a 25th bit is
  `+/- 2^(15+k)` itself, which is a power of two and therefore exact, so `k = 9`
  holds too. It fails at `k = 10`: at `C = 1024` the sum reaches `2^25`, and an
  odd `acc` above `2^24` is not a `float32` at all. **This section states
  `C <= 512` and not `C <= 256`, because 512 is both provable and tight, and a
  bound that is looser than the truth invites the same overstatement back.**
- Past `C = 512`, or at a channel count that is not a power of two, the two arms
  may differ and this port is the MORE accurate of the two, because its sum is
  exact where `np.mean`'s is not. Measured (§4.16.7): worst
  `|ours - long double|` is **0** at `C` in {1, 2, 4, 8, 64, 256, 512} and
  **2.98e-08** — half an `ulp`, so still correctly rounded — at `C` in
  {1024, 2048, 32768}. Agreement with a `float32` arm at `C = 1024` is 126/2000
  for a sequential accumulator and 2000/2000 for `numpy`'s pairwise reduction;
  at `C = 2048`, 78/2000 and 1636/2000; at `C = 32768`, 1439/2000 pairwise. No
  published dots3-note request shape reaches any of it: `C = 1` and `C = 2` are
  what a WAV upload carries.
- **The intermediate type is now GATED and not only derived.**
  `test_dots3_note_audio.cpp`'s *"at 1024 channels a float32 accumulator is
  WRONG, and at 512 it is not"* runs the closed-form near-full-scale pattern
  `32000 + ((c * 7 + f * 37) % 768)` at both counts. At 512 this decoder, the
  long-double reference and an in-test `float32` accumulator agree to the bit;
  at 1024 this decoder is still exact and the `float32` accumulator is
  7.5e-06 away, about 126 `ulp`s. The case asserts BOTH halves, so it cannot
  quietly stop discriminating. What it does NOT gate is `numpy`'s own pairwise
  reduction, which survives at 1024 and needs `C = 4096` to separate; that stays
  measured and unGATED, and a future gate would need a 4096-channel fixture and
  a pairwise reference in the test.

**W7c-1 moved four `audio_processor.cpp` line anchors, and re-pointed them in
the same change.** The shared parser and the new sibling add lines above
`WhisperAudioProcessor`, so §4.14's `:35-79` (the mono decoder), `:91-199` (the
log-mel), `:206-225` (`ExpandAudioPlaceholders`) and `## Owed`'s `:94-101` (A1's
"resample deferred" throw) all shifted. Three of the four moved together and now read `:211-319`, `:326-345` and
`:214-221`, along with the same anchor in
`include/vllm/multimodal/dots3_note_processor.h` and the pad/truncate anchor in
`dots3_note_processor.cpp` (`:228-232`); for those the surrounding prose is
unchanged, because only the line numbers moved. They shifted TWICE — +90 when
W7c-1 inserted the shared walk and the sibling, and +30 again when this repair
wave rewrote the sibling's comment — which is why the values here are the ones
measured against the final file and not against the first insertion.

**The mono decoder is the one that did not merely shift, and the first repair of
it was wrong.** `:35-79` was 45 lines because the decoder carried its own chunk
walk; W7c-1 SPLIT it, so there is no single successor range. It re-points to
`:103-131` — the decoder — over the shared `ParseWavPcm16` at `:49-99`, and
both are named wherever the old range was. The first re-point wrote `:103-151`,
which is neither: at that commit `:151` sat 32 lines inside the mean sibling's
comment block, and the range excluded the parsing the sentence was actually
about. An anchor that a
change falsifies is that change's to repair, and a repair that is not measured
against the file is not a repair.

**The mono entry point is not touched.** `DecodeWavPcm16Mono` keeps its own
`static_cast<float>(s) / 32768.0f` loop verbatim, so `parakeet_transcription.cpp`,
`chat_mm.cpp` and `test_voxtral_e2e.cpp` cannot move by a bit — this slice adds
`DecodeWavPcm16MeanToMono` beside it and only the dots3-note route calls it.
Both share ONE chunk walk (`ParseWavPcm16`), so the multi-channel arm is not a
second hand-written parser. The gate MEASURES the equality rather than asserting
it in prose: a 2-channel buffer whose two channels are equal decodes bit for bit
to the mono decode of one of them.

#### 4.16.3 The reachability proof is an INVERSION, not a new test

The production entry point is unchanged: `ApiServer::handle_chat_completions`
-> `InstallMultiModalChatSeam` (`server_main.cpp:1565`) -> `MakeDots3NoteChatSeam`
-> `RouteDots3NoteAudioWav` (`mm_chat_dots3note.cpp:210`). The smallest failing
test already existed and asserted the OPPOSITE of what this slice makes true:
the subcase *"a STEREO WAV names the container refusal and W7c"* checked
`status == 400` and that the body named W7c. It is replaced by a case that
checks the same request SERVES, which is a true-before / false-after inversion
and is what makes this slice owned rather than merely present.

The replacement does not stop at HTTP 200. The stereo fixture is built as
`L = m + d`, `R = m - d` from two DIFFERENT signals, so its per-sample mean is
EXACTLY `m` — the fixture's own variant 0, the clip every other audio case in
the suite already serves. The case therefore asserts:

1. the stereo request answers 200, with the same prompt-token accounting as the
   mono one, so the placeholder span did not move;
2. the test recomputes `(L[i] + R[i]) / 2` itself, in `int`, and checks it
   equals `m` bit for bit — the independent mean, computed in the test and not
   read out of the production path;
3. the stereo request's first-token logprobs equal the MONO `m` request's;
4. and they DIFFER from the mono `L` request's and from the mono `R` request's.

(4) is what makes (3) load-bearing. A port that took channel 0 would serve `L`
and pass (1) and (3)-shaped equality against nothing; a port that summed without
dividing would serve `2m`. Both are separated by (4), and both are driven as
mutations in §4.16.5.

#### 4.16.4 Three false statements about the oracle, repaired in flow

**`librosa` is not on vLLM's decode or resample path, and this tree said three
times that it was.** Measured at `9035151d6`:
`git grep -n 'librosa' 9035151d6 -- vllm/` returns **three** hits and all three
are comments — `vllm/multimodal/audio.py:31` ("Aligned with
`librosa.get_duration` function"), `vllm/transformers_utils/processors/fireredlid.py:182`
and `vllm/transformers_utils/processors/inkling.py:300` (both naming a
convention). `git grep -n '^\s*import librosa\|^\s*from librosa' 9035151d6`
returns nothing: **the package is never imported**. It appears in
`requirements/test/*` only, as a test dependency.

The real chain, read at the same SHA:

| Stage | What actually runs | `file:line@9035151d6` |
|---|---|---|
| decode, primary | `soundfile` / libsndfile, `f.read(dtype="float32")` | `vllm/multimodal/media/audio.py:29-32`, `:205` |
| decode, fallback | PyAV / FFmpeg, `load_audio_pyav` | `vllm/multimodal/media/audio.py:24-27`, `:47` |
| channel reduction | `np.mean` | `:207-208`, and `:168-169` on the PyAV arm |
| resample | `av.AudioResampler` — **libswresample** through PyAV, and the docstring says so | `vllm/multimodal/audio.py:174-229`, esp. `:180` and `:221` |

Two of the three statements were **production error strings a user reads**, and
both also carried the unresolvable `common/processor.py` anchor §4.16.1
corrects:

- `src/vllm/entrypoints/openai/mm_chat_dots3note.cpp:233-234` — "Upstream
  decodes with librosa through its data parser". Rewritten to name libsndfile
  and PyAV, and NARROWED (§4.16.6).
- `src/vllm/multimodal/dots3_note_processor.cpp:526-528` — "upstream resamples
  in its data parser ... with librosa". Rewritten to name `resample_audio_pyav`
  / libswresample, and re-pointed at W7c-2.
- this spec's `## Owed` — "accepts whatever `librosa` can open". Rewritten to
  say what libsndfile and FFmpeg actually open.

**And one stale BLOCKER in the same `## Owed` entry.** It read "a windowed-sinc
resampler is a numerically delicate port of its own, not a line of glue", which
was written when this tree had no resampler. It has one:
[#2583](https://github.com/mudler/vllm.cpp/issues/2583) landed
`Ltx2ResampleWaveform` (`src/vllm/model_executor/models/ltx2_audio_vae.cpp:1151`,
declared at `include/vllm/model_executor/models/ltx2_audio_vae_encoder.h:190`).
The difficulty claim is still true and the BLOCKER claim is not, so the entry now
says the work is a rate-conversion decision against a seam that exists, owed to
W7c-2, rather than a port that has to be written from nothing.

**One `librosa` mention in this file is left alone, deliberately.**
`src/vllm/multimodal/audio_processor.cpp:215` says a genuine resample would be
"windowed sinc, à la librosa". That describes an ALGORITHM's style, not vLLM's
executing chain, it belongs to the A1 Whisper/Voxtral row rather than this one,
and it is not false. Editing another row's message to satisfy a sweep is how a
correction becomes a drift.

#### 4.16.5 The mutations

Every mutation applied to the tracked source, rebuilt, run, and restored
byte-for-byte, with the binary sha256 and the case counts recorded so a failed
build cannot read as a pass:

| # | Mutation | What a green here would mean |
|---|---|---|
| M1 | take channel 0 instead of the mean | the gate cannot tell a mean from a channel pick |
| M2 | sum the channels without dividing | the gate cannot see a 2x amplitude error |
| M3 | mean over the FRAME axis instead of the channel axis | the gate cannot see the wrong axis, only the wrong shape |
| M4 | delete the production call site — `DecodeWavPcm16MeanToMono` back to `DecodeWavPcm16Mono` in `RouteDots3NoteAudioWav` | the served suite measures a function, not a capability |

Each is applied to the tracked source, rebuilt, run and restored byte for byte,
and each records the binary sha256 AND the case counts, because on this row a
failed build has twice read as a pass and once did so with a binary byte-identical
to a previous mutation's. **The measured table is §4.16.7, written by the
implementation commit.** It is not in this spec commit, because a result written
before it is measured is the defect this protocol exists to prevent.

#### 4.16.6 What is still refused, and to whom it is owed

The refusal stays at the SEAM and stays the SAME predicate as the route, for
the reason W7a recorded and W7b repeated: an `InstallMultiModalChatSeam` throw
is HTTP 400 for one request, while the same throw from inside `encode_mm` sets
`AsyncLLM`'s errored latch and turns every later request, TEXT ones included,
into a 500.

| Refused | Owed to | Where the message is |
|---|---|---|
| any container but RIFF/WAVE PCM16 — `mp3`, `flac`, `ogg`, anything an `input_audio.format` may name | **NOT this row.** The shared codec brick, [#2814](https://github.com/mudler/vllm.cpp/issues/2814): five surfaces want the same demuxer, libsndfile alone reports 26 formats, and no row owns it | `mm_chat_dots3note.cpp`, the decode `catch` |
| a rate that is not `audio_config.sampling_rate` | this row, **W7c-2** | `dots3_note_processor.cpp`, `ProcessWaveform` |
| non-16-bit PCM, a non-PCM `fmt`, a zero channel count, a malformed chunk walk | this row, W7c-2, alongside the rate arm | `audio_processor.cpp`, `ParseWavPcm16` |
| `use_causal`, `use_conv1d_stem` (`use_conv2d_stem = false`), `use_latent_input`, `merge_factor != 1`, a non-`dots` `encoder_type` | this row, unchanged by this slice | `Dots3NoteAudioRefusal` |

The container refusal is NARROWED and not merely reworded: it used to name
"multi-channel or non-16-bit WAV" as owed to W7c, and multi-channel is now
served. Nothing about the container arm is implemented here.

#### 4.16.7 The gate, measured

Built in `/dev/shm` at `-DVLLM_CPP_SERVER=ON -DVLLM_CPP_BUILD_TESTS=ON
-DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`, GCC, `-j 2`. Two suites, by
their real target names:

| Suite | Cases | Assertions |
|---|---|---|
| `test_dots3_note_audio` | **21 / 21 passed** | **3904 / 3904** |
| `test_openai_api_server_dots3_mm_forward` | **27 / 27 passed** | **16344 / 16344** |

`test_dots3_note_audio` sha256
`1814f30286bfd39b49c6ca14c233b4293964a3b2d1fc5a8b682d711965d2678c`;
`test_openai_api_server_dots3_mm_forward` sha256
`042b208fcd41f238548d24760e3f66e170d95a8d17bc56f897c930032cd24420`.

**The RED before, verbatim.** With the tests inverted and `src/` untouched, the
served suite read **27 cases / 25 passed / 2 failed, 16321 assertions / 8
failed**, and the stereo case died at the line that matters:

```text
test_api_server_dots3_mm_forward.cpp:1500: FATAL ERROR: REQUIRE( r.status == 200 ) is NOT correct!
  values: REQUIRE( 400 == 200 )
  logged: stereo body: {"error":{"code":400,"message":"dots3-note audio chat seam:
  this request's audio is not a PCM16 MONO RIFF/WAVE buffer (DecodeWavPcm16Mono:
  not mono). ..."}}
```

The other seven were the three repaired oracle statements, each asserted as a
CLAIM rather than as a word: `#2814` absent, `PCM16 MONO` present,
`multi-channel` present, `librosa` present in the container message; and
`W7c-2` absent, `librosa` present, `libswresample` absent in the rate one.

**The exactness claim of §4.16.2 is MEASURED, not asserted.** Against a
`long double` reference over 517 frames:

| Channels | Worst \|got - ref\| | Exactly equal |
|---|---|---|
| 1 | **0** | 517 / 517 |
| 2 | **0** | 517 / 517 |
| 3 | 1.98682e-08 | 161 / 517 |
| 4 | **0** | 517 / 517 |
| 8 | **0** | 517 / 517 |

Every power of two is exact and 3 is not, which is exactly the shape §4.16.2
predicts. 1.99e-08 is under a third of a float ulp near 1.0.

**That table cannot see the bound, and a repair wave measured where it is.**
Its signal shrinks as `C` grows, because a mean of random samples tends to zero
while the exactness argument is about `|acc|` reaching `32768 * C`. Re-measured
with the samples DRIVEN to near full scale, so `|acc|` runs at its own stated
ceiling, over 2000 draws per channel count against a `long double` reference:

| Channels | Worst \|ours - ref\| | == sequential `float32` | == `numpy` pairwise `float32` |
|---|---|---|---|
| 1, 2, 4, 8, 64, 256, 512 | **0** | 2000 / 2000 | 2000 / 2000 |
| 1024 | 2.980e-08 | 126 / 2000 | 2000 / 2000 |
| 2048 | 2.980e-08 | 78 / 2000 | 1636 / 2000 |
| 32768 | 2.980e-08 | — | 1439 / 2000 |

**So the bit-identity claim holds to `C = 512` and no further**, which is what
§4.16.2 now states. 2.98e-08 is exactly half a `float` ulp at that magnitude, so
this arm is still the correctly-rounded answer where the `float32` arms are not.
The double-rounding sweep is separate and exhaustive rather than sampled: all
**1,300,542,267** `(C, acc)` pairs for every non-power-of-two `C` in `[2, 200]`,
**0** anomalies.

**The intermediate type is GATED, and the gate was proved RED first.** The new
`test_dots3_note_audio.cpp` case runs `32000 + ((c * 7 + f * 37) % 768)` at
`C = 512` and `C = 1024`:

| Channels | this decoder vs `long double` | in-test `float32` accumulator |
|---|---|---|
| 512 | exact, 4 / 4 frames | identical to this decoder, 4 / 4 |
| 1024 | exact, 4 / 4 frames | differs on 4 / 4, worst error 7.51e-06 (~126 ulp) |

**The served case's own numbers.** The stereo fixture's two channels differ
from their mean in **7996 and 7996 of 8000** samples, so a port that picked one
would be visibly wrong. The stereo request's first-token logprobs match the mono
mean request's with a worst gap of **0**, and differ from channel 0's by
**0.2155** and from channel 1's by **0.2828**.

#### The mutation table

Each mutation applied to the tracked source, rebuilt, run, and restored. The
harness asserts that the mutation APPLIED and that the build SUCCEEDED before it
reads any result — **M1 and M2 first came back as build failures**
(`-Werror=unused-variable` on the `denom` the mutation orphaned) and were
reported as `BUILD FAILED -- NOT a test result` rather than as a green, which is
the exact failure mode this row has hit twice.

| # | Mutation | `test_dots3_note_audio` | `test_openai_api_server_dots3_mm_forward` | audio sha256 / served sha256 |
|---|---|---|---|---|
| — | baseline | 21 / 3904 pass | 27 / 16344 pass | `1814f302…` / `042b208f…` |
| M1 | take channel 0 instead of the mean | **1 case / 8 assertions FAILED** | **1 case / 2 FAILED** (`gap_mean == 0.0` and `gap_left > 1e-4`) | `7fb0371d3467a6b456026f5906a4cc89adb089a86de3b8c85a535f0db023dc5c` / `ea0e227123ed1da141c5f7f079b06df875bda1613ba2f66e6d1cca55aa46f08d` |
| M2 | sum the channels without dividing | **1 case / 9 assertions FAILED** | **1 case / 1 FAILED** (`gap_mean == 0.0`) | `d973c90715908c4f4d536eedc1cb591d7b3b628f7078acd29518d9d761291aed` / `fc0e5545b1be80a88588413af8aa8280413064f490549d8aa32a78aecb7a20be` |
| M3 | mean over the FRAME axis, not the channel axis | **1 case / 9 assertions FAILED** | **1 case / 1 FAILED** (`gap_mean == 0.0`) | `f0aeb23884caae649f244139a1481e97bfe9327fda7aa59d7fb1be4c86c49520` / `95bf9489d3b2bf7c02d1af7b2324734f267ac87955b125b91b58f5c241296c67` |
| M4 | delete the production call site (back to `DecodeWavPcm16Mono`) | 21 / 3904 pass | **1 case / 1 FAILED**, `REQUIRE( 400 == 200 )`, and the suite's assertion total drops to 16322 because the FATAL aborts the case | `13b732986cf71a484044c15155e6a6cec3c5aea56cfeef8bd92c77db7eefce4b` / `f24fbb23cd1857f7f76263b3cf59bb299f7b9232588479380980a4cfb12c1a9a` |

Ten distinct shas, none equal to the baseline's or to another row's, so no
result here is a stale binary reporting green.

**M1 and M2 are separated, which is the point of the fixture's construction.**
Both move the answer, and only M1 also fails `gap_left > 1e-4` — because with
channel 0 picked, the served stereo answer IS channel 0's. M2 leaves them
different and fails only the equality. A fixture whose channels were equal, or
whose mean was compared by tolerance alone, could not tell the two defects
apart.

**M4 is the reachability line, and its column shape is the measurement.**
`test_dots3_note_audio` stays fully green under it: the decoder still computes
the right mean, it is simply no longer called. Only the suite that enters
through `ApiServer::handle_chat_completions` moves. That is the difference
AGENTS.md's "Nothing lands dead" asks for, shown rather than argued.

**Restored byte-for-byte, and verified at the BINARY.** After the last mutation
the tree was rebuilt and both binaries hashed again: `1814f302…` and
`042b208f…`, identical to the baseline row above, with both suites green at
21 / 3904 and 27 / 16344. A restored source that produced a different binary
would mean the restore was not byte-for-byte.


### 4.17 W7c-2 RESAMPLES a non-16 kHz WAV, by upstream's own scipy arm

**Issue: [#2828](https://github.com/mudler/vllm.cpp/issues/2828). Brick: W7c-2,
the SAMPLE-RATE half of W7c.** W7c-1 (§4.16) served a multi-channel WAV and was
an exact mirror of a two-line reduction. This half cannot be, and the two were
split so that this one does not borrow that one's credibility.

Before this slice, a PCM16 WAV whose `fmt ` chunk named any rate other than
`audio_config.sampling_rate` was refused with HTTP 400 by
`Dots3NoteAudioProcessor::ProcessWaveform`. After it, the waveform is resampled
to the checkpoint's rate and served, and what the tower sees is
`scipy.signal.resample_poly` at its own defaults over the gcd-reduced ratio.

#### 4.17.1 Why this is a RECORDED DIVERGENCE and not a mirror

Upstream's chain at `9035151d6`, read in `~/_git/vllm`
(`git rev-parse 9035151d6` = `9035151d6c9fb726181469f9e6aa9ccbf9a5dacb`;
`git rev-parse HEAD` = `5559679229bc961848b121ccdeaa8fa5d79bec98`, the parity
pin, which carries no `dots3_note` at all):

| Stage | What runs | `file:line@9035151d6` |
|---|---|---|
| the model selects the target rate | `MultiModalDataParser(target_sr=..., target_channels=1)` | `vllm/models/dots3_note/common/processor.py:523-525` |
| the parser resamples | `self._audio_resampler.resample(...)` | `vllm/multimodal/parse.py:695` |
| the resampler's DEFAULT method | `method: Literal["pyav", "scipy", "soxr"] = "pyav"` | `vllm/multimodal/audio.py:283`, dispatch at `:305-316` |
| what `pyav` is | `av.AudioResampler` — **libswresample** through PyAV/FFmpeg | `vllm/multimodal/audio.py:174-229` |

**libswresample is not bit-identical to itself.** #2828 measured this, and this
slice REPRODUCED it rather than relaying it — ffmpeg 6.1.1-3ubuntu5, one binary,
one input, differing only in CPU dispatch:

```text
ffmpeg -i in.wav -af aresample=16000 -c:a pcm_f32le a.wav
ffmpeg -cpuflags 0 -i in.wav -af aresample=16000 -c:a pcm_f32le b.wav
```

| Run | `identical` | differing samples | worst \|delta\| |
|---|---|---|---|
| #2828's | False | 24691 / 32000 | 9.686e-08 |
| this slice's, on its own 2 s three-tone probe | **False** | **19846 / 32000** | **2.980e-07** |

The counts differ because the probe signals do; the fact does not, and it is the
fact the decision rests on.

A bit-exact gate against upstream's default is therefore impossible **in
principle** and not merely inconvenient. Two further facts point the same way:
its option defaults come from an unpinned linked binary (`av` appears at the pin
only in `requirements/test/cpu.txt`, a TEST lockfile, and PyAV bundles its own
FFmpeg), and the `cutoff` swr auto-resolves is not readable from outside the
source.

**This is the opposite of W6c's precedent, and W6c argues for the split rather
than against it.** PIL's bicubic WAS portable because `Resample.c` is ~200 lines
of documented filter maths over a fixed-point intermediate with exactly one
answer, and §4.13 gated it at 0 of 255 worst absolute difference. libswresample
is float, SIMD-dispatched and option-defaulted. W6c argues FOR porting a
documented filter and AGAINST pretending swresample is one.

#### 4.17.2 What is implemented instead, and why that is still upstream

**`resample_audio_scipy`** (`vllm/multimodal/audio.py:232-250 @ 9035151d6`).
This is not an invented divergence. It is **another arm of upstream's own
switch**, and vLLM already ships that arm in production for another model:
`vllm/model_executor/models/phi4mm.py:580 @ 9035151d6` passes
`audio_resample_method="scipy"`. The `pyav` default is refused here
permanently, for the reason above, and the refusal names it.

**The distance from the real default is measured and recorded here so nobody
re-derives it — AND SO THAT NOBODY QUOTES IT WITHOUT ITS PROBE.** #2828 records
scipy at 51.36 dB from libswresample, soxr at 46.59 and torchaudio at 26.72, on
"band-limited content". This slice re-measured all three against ffmpeg 6.1.1 at
44100 -> 16000, interior only (2000 samples trimmed from each end), and the
ordering holds — but **only on content that reaches the OUTPUT Nyquist**, and
that qualifier is load-bearing:

| Probe (2 s at 44100) | scipy | soxr | torchaudio |
|---|---|---|---|
| 0 -> 7500 Hz sweep | **51.78 dB** | 44.63 | 29.02 |
| noise band-limited to 7500 Hz | **37.47 dB** | 28.19 | 27.08 |
| noise band-limited to 4000 Hz | 61.22 | **96.45** | 62.84 |
| three tones at 440 / 1234 / 3000 Hz | 62.72 | **93.84** | 70.12 |

The first row reproduces #2828's three numbers to within about 2 dB, so
"band-limited content" there means content that FILLS the band up to the new
Nyquist, not content sitting well below it. **On content well below the
transition band the ordering INVERTS and soxr wins by 30 dB**, because a
resampler's transition band cannot matter where there is no energy in it. A
speech encoder sees the first kind. Quoting 51.36 without the probe would make a
signal-dependent measurement read as a property of the algorithms.

**Do NOT reuse `Ltx2ResampleWaveform`**
(`src/vllm/model_executor/models/ltx2_audio_vae.cpp:1151`, landed by
[#2583](https://github.com/mudler/vllm.cpp/issues/2583)). It is a genuine
polyphase resampler and it is the tempting reuse. It is ~25 dB FURTHER from this
oracle on exactly the content a speech encoder sees, because torchaudio's
defaults are a short kernel (`lowpass_filter_width=6`, a Hann window) against
swr's 32-tap kaiser-9. **This slice measured that gap rather than repeating it:
22.8 dB on the 0 -> 7500 Hz sweep (51.78 against 29.02) and 10.4 dB on
band-limited noise**, which is #2828's ~25 dB confirmed. Reaching for it would
trade a 51.78 dB answer for a 29.02 dB one and would look like a simplification.
It is not one, and this paragraph exists so the next reader does not make it.

**And the same caveat applies in the other direction.** On the three-tone probe
torchaudio scores 70.12 dB and BEATS scipy, because the short kernel's poor
transition band never gets exercised. A reviewer who reaches for
`Ltx2ResampleWaveform` and validates it on a low-frequency tone will find it
excellent. The table above is why that would be the wrong probe.

#### 4.17.3 The algorithm, stated exactly, and VERIFIED against scipy's source

A divergence has to name what it does. `resample_poly(x, up, down)` at defaults,
read in the installed **scipy 1.17.1**
(`scipy/signal/_signaltools.py::resample_poly`), not transcribed from the issue:

1. `up = target_sr // gcd`, `down = orig_sr // gcd` — upstream's own reduction
   (`audio.py:244-249`). `resample_poly` then reduces AGAIN by
   `math.gcd(up, down)`, which is a no-op on an already reduced pair.
2. `n_out = ceil(n_in * up / down)`, written upstream as
   `n_out = n_in * up; n_out = n_out // down + bool(n_out % down)`.
3. `max_rate = max(up, down)`; `f_c = 1 / max_rate`; `half_len = 10 * max_rate`;
   `h = firwin(2 * half_len + 1, f_c, window=("kaiser", 5.0))`.
4. `h *= up`.
5. `n_pre_pad = down - (half_len % down)`;
   `n_pre_remove = (half_len + n_pre_pad) // down`; `n_post_pad` is grown until
   `_output_len(len(h) + n_pre_pad + n_post_pad, n_in, up, down) >= n_out +
   n_pre_remove`, where `_output_len(lh, n, up, down) = ((n-1)*up + lh - 1)
   // down + 1`. `h` is zero-padded on both sides by those amounts.
6. `y = upfirdn(h, x, up, down)[n_pre_remove : n_pre_remove + n_out]`.

`firwin` at these arguments (`scipy/signal/_fir_filter_design.py::firwin`,
`fs=None -> 2`, `nyq = 1`, `pass_zero=True`, `scale=True`) reduces to a single
low-pass band `[0, f_c]`, `pass_nyquist` false, so with
`alpha = half_len` and `m = arange(numtaps) - alpha`:

```text
h[i] = f_c * sinc(f_c * m[i]) * kaiser(numtaps, 5.0)[i],  then h /= sum(h)
```

The `scale` step's `scale_frequency` is `0.0` because the first band's left edge
is 0, so `c = cos(0) = 1` and the normalizer is the plain sum. The window is
`scipy.signal.windows.kaiser(numtaps, 5.0, sym=True)`, which is
`i0(beta * sqrt(1 - ((n - alpha)/alpha)^2)) / i0(beta)`.

`upfirdn(h, x, up, down)` is the decimation by `down` of the convolution of `h`
with `x` zero-stuffed by `up`, so

```text
y[i] = sum_j h[i*down - j*up] * x[j]   over every j with 0 <= i*down - j*up < len(h)
```

which the port evaluates over `j` in `[ceil((i*down - len(h) + 1)/up),
floor(i*down/up)]` — about `len(h)/up` terms — rather than over the whole
filter.

**This transcription was CHECKED against scipy rather than trusted.** A
standalone reimplementation of exactly the six steps above, with `i0` as its own
power series `sum_k (x^2/4)^k / (k!)^2`, was run against
`scipy.signal.resample_poly` on `float64` random input for all four rate pairs
this slice gates. Worst `|ours - scipy|` was **6.7e-16 to 1.3e-15** in `double`,
and **0.0** after narrowing both sides to `float32`, at signal scales of 1.8 to
4.1. That is what makes the C++ below a port and not a guess, and the script
that measured it is committed as `scripts/gen-dots3-resample-golden.py`.

#### 4.17.4 The type the arithmetic is done in

`ResampleAudioScipy` takes `float` samples, **designs the filter and accumulates
the convolution in `double`, and narrows once to `float`** at the store.

Upstream's arm narrows earlier: `resample_poly` does `h = xp.asarray(h,
dtype=x.dtype)` BEFORE `h *= up`, so a `float32` input — which is what
`soundfile.read(dtype="float32")` hands it — gets a `float32` filter and a
`float32` accumulation. This port deliberately does not mirror that, and the
reason is that mirroring it is not reproducible: a `float32` accumulation over
the ~56 taps each output sample touches at 44100 -> 16000 depends on summation
order and on whether the compiler contracts a multiply-add, neither of which is
fixed across the platforms this tree builds on. The `double` arm is order-stable
to ~1e-16 relative, which is invisible after the narrowing store.

So the claim this slice makes is bounded, and it is this: **the port equals
`scipy.signal.resample_poly` on a `float64` input, narrowed to `float32`, within
a measured tolerance.** It does NOT claim bit-identity with scipy's own
`float32` arm, and it does not claim anything at all about libswresample beyond
the probe-qualified distances recorded in §4.17.2.

**SAY HOW BIG THAT GAP IS, because it is LARGER THAN THE GATE'S OWN TOLERANCE.**
A reader who is told only that bit-identity is not claimed will infer the
difference sits inside `kResampleTol`. It does not.
`|scipy's real float32 arm − the committed float64 golden|`, both narrowed to
`float32`, measured on scipy 1.17.1 / numpy 2.3.5:

| Case | \|f32 arm − golden\| | against `kResampleTol` = 1.2e-7 |
|---|---|---|
| `Wav44100` | 2.384e-07 | **over** |
| `Wav48000` | 2.384e-07 | **over** |
| `Wav22050` | 1.788e-07 | **over** |
| `Wav8000` | 1.192e-07 | under, by 0.7% |
| `Alias44100` | 1.788e-07 | **over** |

So "just mirror the cast" is not a free repair: narrowing the arithmetic to
`float32` the way `resample_poly` does REDS this gate on four of the five
goldens, and it would red it for the reason §4.17.4 gives rather than for a
defect. Anyone who reaches for that change has to regenerate the goldens from
the `float32` arm in the same commit.

**A PARTIAL mirror is worse than either, and it stays green.** Narrowing only
the taps to `float` and leaving the accumulation in `double` moves the answer
5.96e-08 on four cases and 2.98e-08 on `Alias44100` — under the tolerance
everywhere, so the gate would accept it silently. That is not an argument for
the tolerance being loose. It is what "the gate is a BOUND, not an equality"
means: 1.2e-7 is two `float` ulps at the fixtures' peak, and every arithmetic
that lands inside it is a legitimate one. The defects the gate exists to catch
are orders above it, and §4.17.11's difference table is what says so.

#### 4.17.5 The seam, and why it is opted into PER MODEL

`vllm::multimodal::ResampleAudioScipy` is a shared seam in new files
(`include/vllm/multimodal/audio_resample.h`,
`src/vllm/multimodal/audio_resample.cpp`), beside `audio_processor.cpp` rather
than inside it. **Exactly one caller opts in**: `Dots3NoteAudioProcessor::
ProcessWaveform`.

**A blanket lift would be wrong, because five rows refuse a rate mismatch and
they are not all the same policy.** Parakeet's refusal is upstream-faithful —
`feature_extraction_parakeet.py` raises rather than resampling — while dots3's
upstream resamples. `audio_processor.cpp`'s Whisper/Voxtral refusal
(`:214-221`) is UNTOUCHED here, and so is `parakeet_audio_processor.cpp`. One
row, one model's policy; each of the others lifts its own refusal when someone
reads its own upstream.

**It goes in `ProcessWaveform` and not in `RouteDots3NoteAudioWav`,** even
though upstream resamples in the data parser and not in the processor, because
`ProcessWaveform` is the function that is HANDED a rate and refuses on it. Its
refusal is the one that has to invert. Putting the resample at the route would
leave `ProcessWaveform(x, n, 22050)` still throwing, and a caller reaching the
processor directly — which the front-end suite does — would still be refused.

#### 4.17.6 One correctness defect the lift CREATES, and closes in the same change

`RouteDots3NoteAudioWav` keys the encoder cache on
`proc.HashAudio(decoded.samples, n)`, over the RAW waveform. While every served
rate was 16000 that key was unambiguous. It stops being unambiguous the moment
two rates are served: a file carrying `N` PCM16 samples at 16000 Hz and a file
carrying **the identical `N` samples** at 44100 Hz decode to identical `float`
buffers, hash identically, and must produce DIFFERENT features. `mm_hash` is a
CROSS-REQUEST key — `EncoderCacheManager::cached_` is keyed on it and
`scheduler.cpp:511-590` reuses a hit — so the second request is handed the
first's embeddings.

**Be exact about what that costs, because the obvious phrasing overstates it.**
A collision needs identical raw buffers, and identical buffers at different
rates cannot resample to the same row count, so the observable failure is a
wrong-length splice rather than a quiet substitution. The key is wrong either
way, and a key that is only accidentally caught downstream is not a key.

`Dots3NoteAudioProcessor::HashAudio` therefore gains a three-argument overload
taking the request's `sample_rate`, which hashes the RESAMPLED waveform — the
buffer the tower actually consumes. That makes two requests that resample to the
same waveform share a cache entry, which is correct, and two requests that do
not, not. The two-argument overload stays for the existing callers. The gate is
a direct assertion that the two files above hash differently, and it is proved
RED by handing the three-argument form the raw buffer.

#### 4.17.7 The gate: a CONSISTENCY gate against a stated algorithm

§6.4 option B holds, and this slice's gate is the shape
[#2583](https://github.com/mudler/vllm.cpp/issues/2583) established for exactly
this situation. scipy stands to the vLLM pin as torchaudio stands to the `ltx-2`
pin there: `resample_audio_scipy` is vLLM's OWN code, so the precedent
transfers.

**Say plainly what it is.** This gate establishes that the port computes
`scipy.signal.resample_poly` at scipy's defaults. It does NOT establish parity
with upstream's `pyav` default, which §4.17.1 shows cannot be gated by anyone,
and it does not establish that scipy's answer is the one dots3-note was trained
against. What is claimable is that the port implements an arm vLLM ships, and
that it implements it correctly.

Committed goldens, generated by `scripts/gen-dots3-resample-golden.py` against
scipy 1.17.1 into `tests/vllm/models/dots3_note_resample_golden.h`, for the four
rate pairs the issue names — 44100, 48000, 22050 and 8000 -> 16000 — plus a
FIFTH case at 44100 -> 16000 on a signal carrying a tone ABOVE the 8 kHz output
Nyquist, which is the only content that can separate a real anti-alias filter
from picking samples.

**A tolerance alone gates nothing here**, because a resampler that returns its
input, or returns zeros, passes one. Four assertions per case, and each names
the defect it excludes:

| Assertion | What it excludes |
|---|---|
| `out.size() == ceil(n * target / orig)` | returning the input unresampled; an off-by-one in `n_out` |
| `max abs(out) > lo` for a stated `lo` | returning zeros, or a filter whose normalization collapsed |
| `max abs(out - golden) <= tol`, `tol` ~ 2x the measured float floor | every value defect, including the `n_pre_remove` centring |
| `max abs(golden - naive) > sep`, `naive` computed IN THE TEST | that the tolerance above is DISCRIMINATING at all: on the alias case a nearest-sample decimation is far from the golden, and the margin is printed |

`ref_resample` is a FOURTH reference namespace under W7a's existing run-time
enumeration instrument, not a second instrument: W7b added `ref_chunks` the same
way, and reference code the instrument does not read is reference code whose
`std::`-only independence nothing measures. Its `kDistinctQualifiedNames` and
`kQualifiedNameOccurrences` are the instrument's own output and are recorded in
§4.17.9 by the implementation commit.

#### 4.17.8 Reachability — TWO refusal cases INVERT

The production entry point is unchanged: `ApiServer::handle_chat_completions`
-> `InstallMultiModalChatSeam` -> `MakeDots3NoteChatSeam` ->
`RouteDots3NoteAudioWav` -> `Dots3NoteAudioProcessor::ProcessWaveform`. The
smallest failing tests already exist and assert the OPPOSITE of what this slice
makes true. That true-before / false-after is the ownership test:

- `tests/vllm/entrypoints/openai/test_api_server_dots3_mm_forward.cpp:1397`,
  *"a 22050 Hz WAV names the resampler refusal and W7c-2"*, which checks
  `status == 400`;
- `tests/vllm/models/test_dots3_note_audio.cpp:820`, *"a rate that is not
  `audio_config.sampling_rate` names W7c"*, which checks the throw's message.

**Both anchors had MOVED from the numbers #2828 records** (`:1389` and `:818`),
by 8 and 2 lines, because W7c-1 landed between the issue being written and this
slice starting. They are measured at this head, and this row has had anchors go
stale inside a single pull request before.

The served replacement does not stop at HTTP 200. The 44.1 kHz fixture is
22050 samples, so `ceil(22050 * 160 / 441)` is exactly 8000 — the same length as
the mono clip every other audio case serves, and therefore the same 7-token
placeholder span. **That token count is the assertion that a no-op resample
cannot survive**: an unresampled 22050-sample waveform expands
`ceil(22050 / 1280)` = 18 placeholders, not 7. It is checked, and it is
independent of the resampler's values.

Then the equality: the 44.1 kHz request's first-token logprobs against the same
audio resampled offline and served at 16 kHz, and — the assertion that makes
that one load-bearing — DIFFERING from the same 22050 samples served
mislabelled as 16 kHz.

#### 4.17.9 The mutations

Each applied to the tracked source, rebuilt, run, and restored byte-for-byte,
with the binary sha256 AND the case counts recorded, because on this row a
failed build has twice read as a pass:

| # | Mutation | What a green here would mean |
|---|---|---|
| M1 | return the input unresampled | the gate cannot tell a resample from a pass-through |
| M2 | return zeros of the right length | the gate is a shape check wearing a tolerance |
| M3 | drop the anti-alias filter — decimate by picking samples | the gate cannot see aliasing, only interpolation |
| M4 | off-by-one the `n_pre_remove` centring | the gate cannot see a one-sample phase shift |
| M5 | delete the production call site — the resample in `ProcessWaveform` back to the throw | the served suite measures a function, not a capability |
| M6 | hash the RAW waveform in the three-argument `HashAudio` | §4.17.6's cache collision is back and nothing sees it |
| M7 | the ROUTE reverts to the two-argument `HashAudio` | the overload computes the right key and nothing asks for it |
| M8 | the route hands the RAW buffer over as the resample "answer" (PR #2842 F2) | the shared-buffer argument is trusted and never checked |
| M9 | `kMaxUpsampleRatio` is widened past the attack (PR #2842 F2) | the output bound is decorative |
| M10 | delete the production call site — the resample in `ProcessWaveform` | the served suite measures a function, not a capability |
| M11 | the route stops handing the buffer over, so it resamples TWICE (PR #2842 F2) | *expected GREEN* — see §4.17.14 |

The measured table is §4.17.11, written by the implementation commit. It is not
in this spec commit, because a result written before it is measured is the
defect this protocol exists to prevent.

#### 4.17.10 What is still refused, and to whom it is owed

The refusal stays at the same call path as the route, for the reason W7a
recorded and W7b and W7c-1 repeated: an `InstallMultiModalChatSeam` throw is
HTTP 400 for one request, while the same throw from inside `encode_mm` sets
`AsyncLLM`'s errored latch and turns every later request, TEXT ones included,
into a 500.

| Refused | Owed to | Where |
|---|---|---|
| any container but RIFF/WAVE PCM16 | **NOT this row.** The shared codec brick, [#2814](https://github.com/mudler/vllm.cpp/issues/2814) | `mm_chat_dots3note.cpp`, the decode `catch` |
| non-16-bit PCM, a non-PCM `fmt`, a zero channel count, a malformed chunk walk | this row | `audio_processor.cpp`, `ParseWavPcm16` |
| a non-positive sample rate | this row | `ResampleAudioScipy` |
| a reduced ratio whose `max(up, down)` exceeds `kMaxPolyphaseRate` | this row, and it is a **DIVERGENCE** — see below | `ResampleAudioScipy` |
| a reduced `up/down` above `kMaxUpsampleRatio` | this row, and it is a **DIVERGENCE** — see below | `ResampleAudioScipy` |
| `use_causal`, `use_conv1d_stem` (`use_conv2d_stem = false`), `use_latent_input`, `merge_factor != 1`, a non-`dots` `encoder_type` | this row, unchanged by this slice | `Dots3NoteAudioRefusal` |
| upstream's `pyav`/libswresample arm, permanently | nobody — §4.17.1 is the reason, and it does not expire | this section |

**The ratio bound is a deliberate divergence and is recorded as one.** The
filter is `20 * max(up, down) + 1` taps, and `max(up, down)` is set by the
REQUEST, because a WAV's `fmt ` chunk names its own rate. A request declaring
999983 Hz reduces to `max(up, down) = 999983` and asks this process for a
20-million-tap filter and 20 million Bessel evaluations before any audio is
touched. Upstream has no such guard and would do the same thing; upstream is
also not the surface this refusal protects. `kMaxPolyphaseRate = 100000` caps
the design at ~2M taps / ~16 MB, and every real rate reduces far below it
(44100 -> 441, 48000 -> 3, 22050 -> 441, 8000 -> 2; even a coprime 44101 Hz
gives 44101). The refusal names the bound, the reason, and that upstream has
none. It is gated in both directions: a rate just past it refuses, and 44.1 kHz
serves.

**The filter bound is not an output bound, and the gap was a denial of service
(finding F2 of the fresh review of [PR #2842](https://github.com/mudler/vllm.cpp/pull/2842)).** `up` is
`target_sr / gcd`, so on a 16 kHz target it can never exceed 16000. A `fmt `
chunk declaring 1 Hz therefore reduces to `up/down = 16000/1`: `max(up, down)`
is 16000, it sails under the 100000 filter bound, it designs a cheap filter —
and then it asks for sixteen thousand output samples per input sample. Measured
against `libvllm.a` at `0c440b6c3`:

| `orig_sr` | `n_in` | verdict | `n_out` | wall |
|---|---|---|---|---|
| 1 | 20000 | ACCEPTED | 320000000 (1220.7 MB) | 2.301 s |
| 2 | 40000 | ACCEPTED | 320000000 (1220.7 MB) | 2.439 s |
| 999983 | 1000 | REFUSED (`kMaxPolyphaseRate`) | — | — |

Under `ulimit -v 900000` the 1 Hz call threw `std::bad_alloc`, which is a bare
`std::exception` and NOT `InputValidationError`, so the server answered **HTTP
500 for a property of the REQUEST** — exactly what the table above and the
route's own comment say must not happen. `ParseWavPcm16` applies no rate floor
and W7b lifted the length ceiling, so nothing upstream of the seam bounded it.
Before W7c-2 the path did not exist, because every rate but 16000 was a 400: the
guard closes a regression W7c-2 introduced.

**The second bound is on the RATIO, and it has to be.** `up` alone cannot
separate the two cases: a coprime 44101 Hz is a DOWNsample this seam serves, and
gates that it serves, and it also reduces to `up = 16000`. `up/down` separates
them completely — 0.363 against 16000. Bounding the ratio also never refuses a
long clip, because it bounds the output as a multiple of an input the client
already paid to upload, rather than as an absolute length.

`kMaxUpsampleRatio = 8` is **four times the largest ratio this row serves**. The
highest real upsample into 16 kHz is telephony's `8000 -> 16000 = 2`; 11025
gives 1.451, and every rate at or above the target gives less than 1. The bound
admits any source rate down to 2000 Hz on a 16 kHz target and an 8 kHz source on
a 48 kHz one, and it caps the seam's allocation at 32 bytes per input sample. It
is gated in both directions ONE HERTZ APART, at the unit seam and over HTTP:
2000 Hz reduces to 8/1 and serves, 1999 Hz is coprime with 16000, reduces to
16000/1999 = 8.004, and refuses. The served case also asserts the body does NOT
name §4.15.3's chunk refusal, which is what separates "refused" from "refused
after allocating 16000000 samples" — that is precisely what the RED-before did.


#### 4.17.11 The gate, measured

**These are W7c-2's LANDING numbers, at `0c440b6c3`.** §4.17.14 carries the ones
after #2842's repair; both are kept, because a mutation table whose arms were
measured against a different binary from its baseline is not a mutation table.

Built in `/dev/shm` at `-DVLLM_CPP_SERVER=ON -DVLLM_CPP_BUILD_TESTS=ON
-DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`, GCC, `-j 2`. Two suites, by
their real target names:

| Suite | Cases | Assertions |
|---|---|---|
| `test_dots3_note_audio` | **24 / 24 passed** | **3992 / 3992** |
| `test_openai_api_server_dots3_mm_forward` | **28 / 28 passed** | **16374 / 16374** |

`test_dots3_note_audio` sha256
`db03ea5e790307297836827d6b0ce41fd0b095e70f8981100f87ab9ae9d58c8e`;
`test_openai_api_server_dots3_mm_forward` sha256
`e0158068e12a18708b6cd4978eb2d6c25ed86cddaea26c6624f1ffcf45b4a157`.

**Both shas moved twice during the slice, and the whole sweep was re-run each
time rather than patched.** Once when §4.17.13's two outside controls were added
to the served case, and once when `ResampleAudioScipyOutputLength` was deleted
for being reachable only from a test. A mutation table whose arms were measured
against a different binary from its baseline is not a mutation table.

**The port reproduces scipy BIT FOR BIT on four of the five golden cases.**

| Case | Conversion | `n_in` -> `n_out` | peak | worst \|ours - scipy\| |
|---|---|---|---|---|
| `Wav44100` | 44100 -> 16000 | 441 -> 160 | 0.98991 | **0** |
| `Wav48000` | 48000 -> 16000 | 480 -> 160 | 0.98984 | **0** |
| `Wav22050` | 22050 -> 16000 | 220 -> 160 | 0.98989 | **0** |
| `Wav8000` | 8000 -> 16000 | 80 -> 160 | 0.99038 | **0** |
| `Alias44100` | 44100 -> 16000 | 441 -> 160 | 0.50021 | **8.31e-18** |

Worst against `ref_resample`, the independent second transcription, is
**2.95e-08** — half a `float` ulp, so the two disagree only in the narrowing
store.

**THE TOLERANCE IS NOT TWO TIMES THAT FLOOR, AND SAYING WHY IS THE POINT.**
"~2x the measured float floor" would be 1.7e-17, and gating there would be
gating on a coincidence of rounding rather than on a bound: the port and scipy
agree to ~1e-15 in DOUBLE, and everything after that is one narrowing store
whose granularity is a `float` ulp. A legitimate platform difference — another
libm's `sin` by one ulp in the filter taps, or a contracted multiply-add in the
convolution — moves the double answer by ~1e-16 relative, which usually narrows
to the same `float` and can narrow to the adjacent one. **The gate is therefore
TWO FLOAT ULPS at the fixtures' peak of ~0.99, which is 1.2e-7**, and every
defect it exists to catch is orders above it.

**The difference assertion, which is what makes that tolerance mean anything.**
A nearest-sample decimation computed in the test sits this far from the golden:

| Case | nearest-sample vs golden | against a tolerance of |
|---|---|---|
| `Wav44100` | 0.0538 | 1.2e-7 |
| `Wav48000` | 0.0530 | 1.2e-7 |
| `Wav22050` | 0.1480 | 1.2e-7 |
| `Wav8000` | 0.3495 | 1.2e-7 |
| `Alias44100` | 0.4568 | 1.2e-7 |

**The generator's own agreement, which is what makes the C++ a port.** The six
steps written a second time in plain Python, with `i0` as its own power series,
against `scipy.signal.resample_poly` itself: worst 6.11e-15 in `double` over the
five cases, and **0.0** after narrowing both sides to `float32` on four of them
(2.75e-18 on `Alias44100`). scipy 1.17.1, numpy 2.3.5.

**The independence instrument's counts, from its own reading of the file.**

| Namespace | Distinct | Occurrences | Scopes |
|---|---|---|---|
| `ref_front` | 11 | 71 | `std` |
| `ref_tower` | 6 | 53 | `std` |
| `ref_chunks` | 2 | 25 | `std` |
| `ref_resample` | **5** | **63** | **`std`** |

`ref_resample` reaches exactly two transcendentals, `std::sin` and `std::sqrt`,
beside `std::int64_t` (39), `std::size_t` (14) and `std::vector` (8). The
instrument's own standing pp-number gate is unchanged and still green.

**The front end's own numbers.** At 22050 -> 16000 the front end's mel is
BIT-IDENTICAL to the mel of the same waveform pre-resampled and handed in at
16000: **0 of 1600 values differ**. The resample itself is 2.98e-08 from
`ref_resample` there.

**The served case's numbers.**

| Comparison | Worst first-token logprob gap |
|---|---|
| 44.1 kHz vs its OWN offline resample, through a PCM16 container | 0.00957 |
| 44.1 kHz vs the NATIVE 16 kHz recording of the same signal | **0.00705** |
| 44.1 kHz vs SILENCE | 0.2488 |
| 44.1 kHz vs a DIFFERENT clip | 0.2605 |

The second row is the one worth reading twice: the resampled 44.1 kHz clip lands
CLOSER to a native 16 kHz recording of the same continuous signal than to its own
offline resample, because the offline arm pays a PCM16 quantization the served
arm does not.

#### 4.17.12 The RED before, verbatim, and one defect it found

With `src/` changed and the tests still asserting the refusals, the front-end
suite read **21 cases / 20 passed / 1 failed, 3914 assertions / 4 failed**:

```text
test_dots3_note_audio.cpp:827: ERROR: CHECK( msg.find("22050") != std::string::npos ) is NOT correct!
  values: CHECK( 18446744073709551615 != 18446744073709551615 )
```

with `W7c-2`, `RESAMPLING IS NOT PORTED` and `libswresample` failing the same
way: the throw those four asserted on no longer happens.

The served suite read **27 cases / 26 passed / 1 failed, 16344 assertions /
5 failed**, and it did NOT read what a correct implementation would have made it
read:

```text
test_api_server_dots3_mm_forward.cpp:1403: ERROR: CHECK( r.status == 400 ) is NOT correct!
  values: CHECK( 500 == 400 )
  logged: body: {"error":{"code":500,"message":"WhisperAudioProcessor: resample
  deferred; provide audio at cfg.sampling_rate (16 kHz)","param":null,
  "type":"InternalServerError"}}
```

**That 500 is a real defect the served inversion found, and a unit test on the
resampler could not have.** `ProcessWaveform` rebound the sample pointer and the
length after resampling and left `sample_rate` at the REQUEST's value, so the
resampled buffer was still described as 22050 Hz when it reached
`WhisperAudioProcessor::ProcessWaveform` — which this drives once per chunk, and
which carries its OWN rate refusal for the Whisper/Voxtral row
(`audio_processor.cpp:214-221`). That refusal is a bare `runtime_error`, so the
server answered HTTP 500 rather than 400. Before W7c-2 the missing assignment was
unreachable, because the refusal above it guaranteed the two rates were equal.
The three variables that describe a waveform now move together.

#### 4.17.13 The mutation table, measured

Each mutation applied to the tracked source, rebuilt, run, and restored. The
harness asserts that the mutation APPLIED and that the build SUCCEEDED before it
reads any result, and records the binary sha256 for every arm, because on this
row a failed build has twice read as a pass.

| # | Mutation | `test_dots3_note_audio` | `test_openai_api_server_dots3_mm_forward` |
|---|---|---|---|
| — | baseline | 24 / 3992 pass | 28 / 16374 pass |
| M1 | return the input unresampled | **2 cases / 5 FAILED**, `kw.num_samples == kAudioSamples` | **2 cases / 3 FAILED**, `t22 == t16 - 2` |
| M2 | return zeros of the right length | **2 cases / 16 FAILED**, `worst <= kResampleTol` | **1 case / 2 FAILED**, `gap_native < 5e-2` |
| M3 | drop the anti-alias filter, decimate by picking samples | **2 cases / 11 FAILED**, `worst <= kResampleTol` | **1 case / 1 FAILED**, `gap_native < 5e-2` |
| M4 | off-by-one the `n_pre_remove` centring | **2 cases / 11 FAILED**, `worst <= kResampleTol` | 28 / 16374 pass |
| M5 | delete the production call site | **1 case FAILED**, the subcase THREW | **2 cases / 2 FAILED**, `r.status == 200` |
| M6 | hash the RAW waveform in the 3-argument `HashAudio` | **1 case / 1 FAILED**, `at_target != at_44100` | **1 case / 1 FAILED**, `at16.status == 200` |
| M7 | the ROUTE reverts to the 2-argument `HashAudio` | 24 / 3992 pass | **1 case / 1 FAILED**, `at16.status == 200` |

Binary sha256 prefixes, none equal to the baseline's on the suite the mutation
can reach: M1 `4bf541b1…` / `ce779bdf…`; M2 `e0dffd57…` / `86f2713b…`; M3
`6f709b05…` / `62557952…`; M4 `28ac7a83…` / `64ed66d4…`; M5 `a8e5ca14…` /
`95fa8d03…`; M6 `0af5171a…` / `49c69922…`; M7 `dab4a93c…` / `995f33b5…`.
Fourteen distinct values, none equal to the baseline's `db03ea5e…` /
`e0158068…`.

**M4 is the one the served suite cannot see, and that is stated rather than
hidden.** A one-sample phase shift moves the front-end comparison by far more
than 1.2e-7 and moves the served answer by less than the 5e-2 the native-clip
control allows. The value gate is the front-end suite's; the served suite gates
that the capability is REACHED and that it is not dead or aliased.

**M2 and M3 first read GREEN on the served suite, and that finding changed the
test rather than the report.** Its value assertion compared the request against
an offline reference computed with the SAME production code, so both sides moved
together — a shared-helper consistency check wearing a correctness gate. Two
controls now come from outside the resample path: a WAV of literal silence,
which is never resampled at all, and the natively-16 kHz fixture, which is the
same continuous signal from the same closed form. §4.17.11 carries their
numbers.

**M7 is the reachability line for §4.17.6, and it works for a reason worth
recording.** Reverting the ROUTE to the two-argument hash leaves the front-end
suite fully green — the overload still computes the right key, it is simply not
asked for — and reds only the suite that enters through
`ApiServer::handle_chat_completions`. It reds there because the inverted subcase
serves the SAME 8000-frame buffer at 22050 Hz and then at 16000 Hz on ONE
harness, so the collision §4.17.6 describes is a live cross-request cache hit and
not a hypothetical.

**Restored byte-for-byte, and verified at the BINARY.** After the last mutation
the tree was rebuilt and both binaries hashed again: `db03ea5e…` and
`e0158068…`, identical to the baseline row, with both suites green at 24 / 3992
and 28 / 16374.


#### 4.17.14 The PR #2842 fresh-review repair, measured

Two fresh-review findings, repaired on `row/MODEL-MM-DOTS3-NOTE-W7C2` on top of
`0c440b6c3`. Same recipe as §4.17.11 — `/dev/shm`, `-DVLLM_CPP_SERVER=ON
-DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`, GCC,
`-j 2` — on a DIFFERENT host and build directory, so the baseline shas below are
not §4.17.11's even though the source is byte-identical. This build is not
byte-reproducible across checkouts and never claimed to be; what a sha proves
here is that two arms of THIS sweep are different binaries.

| Arm | `test_dots3_note_audio` | `test_openai_api_server_dots3_mm_forward` |
|---|---|---|
| baseline, `0c440b6c3` | `b181a5dd…` 24 / 3992 pass | `8c117844…` 28 / 16374 pass |
| **RED**, the new cases only | `356e5b48…` **23 / 24 cases, 13 of 4008 FAILED** | `213f7c9e…` **27 / 28 cases, 6 of 16385 FAILED** |
| **GREEN**, repaired | `5c583c69…` 24 / 4014 pass | `99485ce5…` 28 / 16385 pass |

**What the RED said, which is the whole finding.** The served 1 Hz request DID
answer 400 — for the wrong reason and far too late. Its body was §4.15.3's
`this request's 16000000 samples need 1000 chunks`, which is thrown AFTER the
resample: the process had already built the sixteen-million-sample buffer that
the finding is about. The subcase's `r.body.find("chunks") == npos` assertion is
the one that separates "refused" from "refused after allocating", and it is the
assertion the fix turns.

| # | Mutation | `test_dots3_note_audio` | `test_openai_api_server_dots3_mm_forward` |
|---|---|---|---|
| M6 | hash the RAW waveform in the 3-argument `HashAudio` | `64adbaf0…` **1 case / 1 FAILED** | `1664dcff…` **1 case / 1 FAILED** |
| M7 | the ROUTE reverts to the 2-argument `HashAudio` | `19be9058…` 24 / 4014 pass | `8725ae62…` **1 case / 1 FAILED** |
| M8 | the route hands the RAW buffer over as the "answer" | `d6fdd6d5…` 24 / 4014 pass | `76fc721f…` **1 case / 1 FAILED** |
| M9 | `kMaxUpsampleRatio` widened to 100000 | `4793b85a…` **1 case / 13 FAILED** | `957f3d2e…` **1 case / 6 FAILED** |
| M10 | delete the production resample call site | `843fb58d…` **1 case FAILED** (it THREW) | `3fd981db…` **2 cases / 2 FAILED** |
| M11 | the route stops handing the buffer over | `fb5b254c…` 24 / 4014 pass | `b0dc8199…` 28 / 16385 pass |

Twelve distinct shas, none equal to the green row's `5c583c69…` / `99485ce5…`.
M10's unit arm is the doctest shape where a THROWN case reports zero failed
assertions and one failed case, so the case count is what reads it; `rc` was 1.

**M11 IS GREEN ON PURPOSE, and saying so is the point.** Resampling once instead
of twice is behaviour-preserving: the same key, the same features, the same
answer. No test can see it, and inventing an instrument that counts resamples
would be another `ResampleAudioScipyOutputLength` — a symbol reachable only from
a test. What IS gated is the hazard the shared buffer introduces, which is a
caller handing over the WRONG buffer: M8 covers that, and the unit subcase
asserts that handing the buffer over and rebuilding it produce the same key.
M6 and M7 confirm §4.17.6's contract survived the change.

**Restored byte-for-byte, and verified at the BINARY.** After the last mutation
the tree was rebuilt and both binaries hashed again:
`5c583c698f9a54f09fd40b66300bad6129ed96b50e8f10097089e5d7aa6d2e7d` and
`99485ce5ca1ceff1600f31029349fb9af7abd5125bc0a4b0233508c911e2d887`, identical to
the green row, with both suites at 24 / 4014 and 28 / 16385.

**RE-MEASURED AFTER MERGING `origin/main` `73db7a8a3`, because a merge can
falsify prose the code still supports.** Both binaries moved — main brought other
changes into `libvllm.a` — and both suites are green at the same counts:
`test_dots3_note_audio` `e034b19f…` 24 / 4014, and
`test_openai_api_server_dots3_mm_forward` `3fd8caced…` 28 / 16385. The mutation
sweep above ran on the PRE-merge binaries, which is why its baseline row names
`5c583c69…` / `99485ce5…` and not these.

**`test_parakeet_audio_processor` is untouched and stays 6 / 41054.** Parakeet's
rate refusal is upstream-faithful — `feature_extraction_parakeet.py` raises
rather than resampling — and neither finding reaches it.


### 4.18 W8a applies EVERY modality in ONE pass, so one request carries TWO features

**Issue: [#2860](https://github.com/mudler/vllm.cpp/issues/2860). Brick: W8a,
the multi-item, multi-modality half of W8.** Before this slice the dots3-note
chat seam located exactly ONE image part and exactly ONE audio part, declared
`{"image": 1, "audio": 1}`, and refused a request carrying both BY NAME with
HTTP 400. After it the seam serves any mix, declares `{"image": 512,
"audio": 128}`, and a single request carries as many `mm_features` as it has
media parts.

**This is the first request in this repository to carry more than one
`mm_feature`.** Every production seam emitted exactly one — `chat_mm.cpp:157`
and `:184` for Qwen3-VL and Whisper, `mm_chat_dots3note.cpp:185` and `:320`
here — and every served test asserted one. The machinery below the seam is
N-generic in SHAPE and had never been exercised:
`MultiModalInputs::mm_features` is a vector documented "one per placeholder
item" (`include/vllm/multimodal/inputs.h:133`); `try_schedule_encoder_inputs`
walks `GetMmFeaturesInWindow` over the request's item list
(`scheduler.cpp:495-563`); `execute_mm_encoder` loops the scheduler's per-request
input ids and dispatches `EncodeMm` per item (`runner.cpp:1955-1978`);
`gather_mm_embeddings` slices one encoder output per overlapping item and marks
each span in `is_mm_embed` (`runner.cpp:1999-2089`); and
`EmbedMmDots3NoteForCausalLM` scatters `*inputs.mm_embeds` "concatenated in mask
order" (`dots3_note_registry.cpp:358-470`, the mask-order loop at `:408-413`). **Proving it carries N is the
work, and §4.18.7 records that it did on the first try.**

#### 4.18.1 Why the two expanders could not simply both run

`ExpandImagePlaceholders` (`qwen3vl_processor.cpp:175-206`) and
`ExpandAudioPlaceholders` (`audio_processor.cpp:326-345`) each REBUILD the whole
id vector and report offsets into the vector THEY built. Running them in
sequence over one prompt therefore measures the second one's offsets against the
first one's UN-expanded input: on the fixture prompt
`[<|audio_comp_start|>, <|audio_comp_pad|>, <|audio_comp_end|>, "hello",
<|img|>, <|imgpad|>, <|endofimg|>]` the audio pass expands its pad to seven and
reports `[1, 7]`, and an image pass over that OUTPUT is correct — but an image
pass over the ORIGINAL reports `[5, 4]` where the true position is `[11, 4]`,
six rows early, straddling the audio span. Nothing downstream can detect that:
the counts still balance, `n_rows == n_masked` still holds, and the answer is
confidently wrong. That is why the pre-W8a seam refused instead of chaining, and
why the mutation in §4.18.8 that restores the chaining has to be caught by a
LOGPROB assertion rather than a status or a token count.

#### 4.18.2 Upstream's shape, and what this port mirrors

Read in `~/_git/vllm`, `git rev-parse HEAD` = `5559679229bc961848b121ccdeaa8fa5d79bec98`
— the parity pin, which carries no `dots3_note` at all — with the sources at
`git rev-parse 9035151d6` = `9035151d6c9fb726181469f9e6aa9ccbf9a5dacb`.

| What | Upstream | `file:line@9035151d6` |
|---|---|---|
| the per-modality rule | `PromptReplacement(modality, target, replacement)` | `vllm/multimodal/processing/processor.py:423-519` |
| the image rule's target | `[image_start_id, image_pad_id, image_end_id]` | `vllm/models/dots3_note/common/processor.py:735-756` |
| the audio rule's target | `[audio_start_id, audio_pad_id, audio_end_id]` | `common/processor.py:757-783` |
| the replacement content | `PromptUpdateDetails.select_token_id(full, pad_id)` | `processing/processor.py:206-256` |
| the LIST of rules, built per request | `updates: list[PromptUpdate]` | `common/processor.py:725-812` |
| ONE pass over the id stream | `apply_token_matches` -> `_apply_matches` -> `_plan_prompt_updates` | `processing/processor.py:944-957`, `:906-941`, `:799-857` |
| target matching | `iter_token_matches`, non-overlapping | `processing/processor.py:619-657` |
| the declared limits | `{"image": 512, "video": 1, "audio": 128}` | `common/processor.py:527-534` |

`_plan_prompt_updates` is general over INSERT and REPLACE modes and over empty
targets. Both dots3-note rules are REPLACE with a non-empty three-id target, and
on that subset the planner reduces exactly to: repeatedly find, for each
modality queue that still has items, the FIRST occurrence of its target at or
after `prev_end_idx`; apply the earliest match, breaking a tie by the modality's
position in the list (`min(..., key=lambda item: (item[1], _next_priority(item[0])))`
at `:871-874`, over the queue priority `_next_priority` returns at `:794-797`);
set `prev_end_idx` to that match's end. `ApplyPromptReplacements`
(`src/vllm/multimodal/processing/processor.cpp`) is that reduction, and the
narrowing is recorded in the header rather than implied: an INSERT-mode rule and
an empty target are refused BY NAME, because a port that silently treated them
as REPLACE would answer 200 with the wrong id stream.

#### 4.18.3 The triple is the key, and that is a behaviour change

The old expanders key on the PAD id ALONE. The new applier keys on the whole
`[start, pad, end]` target, which is upstream's own key. Two consequences, both
upstream's:

- A bare `<|imgpad|>` typed into a user's TEXT is no longer expanded. The old
  `ExpandImagePlaceholders` matched it and then threw
  `"more image placeholders than grids"`, which reached the client as a 400
  naming an internal helper. It is now an ordinary token the embedding table
  looks up, exactly as upstream leaves it.
- A user who types the WHOLE triple before their real image part takes that
  item's grid. Upstream does the same thing, for the same reason: the target is
  the only thing either side has to go on. Mirrored rather than guarded.

What is NOT relaxed is the item count. If a rule's items are not all consumed by
the end of the pass, `ApplyPromptReplacements` throws BY NAME with the modality,
the count found and the count expected. Upstream reaches the same conclusion
through `_all_items_found` (`processing/processor.py:896-903`); dropping an item
silently is the one outcome that produces a fluent wrong answer.

#### 4.18.4 The ORDER of `mm_features` is load-bearing, and it is the stream's

`GetMmFeaturesInWindow` (`utils.cpp:9-50`) is a pair of BINARY SEARCHES over
`offset` and over `offset + length`. Both the scheduler and the runner call it.
A feature list that is not sorted ascending by `offset` makes both searches
return a window that silently omits an item, and the runner then reports an
encoder-cache miss or scatters the wrong rows. The one-pass walk emits spans in
ID-STREAM order by construction, which is the sorted order, and the seam pushes
`mm_features` in exactly that order — modality is a FIELD of the span, never the
loop that produces it. A per-modality outer loop would be the natural way to
write this and would be wrong; §4.18.8's M2 is that mistake, and it reds.

#### 4.18.5 The declared limits, and why `video` stays ABSENT

`Dots3NoteChatSupportedMmLimits` now returns `{"image": 512}` and, when the
install built an audio tower, `{"audio": 128}`. Those are upstream's own numbers
(`common/processor.py:530`, `:533`). `video` is NOT declared, although upstream
declares `{"video": 1}` beside them, and the omission is the point:
`BaseProcessingInfo` reads an ABSENT modality as limit 0
(`context.py:414-415`), so the entrypoint refuses a video part with upstream's
own `"At most 0 video(s) may be provided in one prompt."` — byte for byte the
refusal this seam already produced. Declaring `{"video": 1}` here would promise
a capability §4.18 does not build and the tower cannot serve; the seam's ceiling
has to be what it can actually build, which is the rule the `has_audio`
parameter has encoded since W7a.

The limits are the OTHER operand of a `min()` fold with the engine's
`--limit-mm-per-prompt` (`context.py:392-405`), so a user limit can still only
LOWER them. What changed is that lowering is now the only way to get the old
`{"image": 1, "audio": 1}` behaviour back.

#### 4.18.6 Gate form: §6.4 option B, and the instrument is EXTENDED not copied

No oracle. ~290 GB fp8 against a 119-122 GiB ceiling on every host this project
reaches, and the pin carries no `dots3_note` at all, so there is nothing to run
the same workload on. Correctness is argued by an **independent in-test
reference sharing no helper with the implementation**, and its independence is
MEASURED rather than asserted: `test_dots3_note_audio.cpp` already carries an
enumeration instrument that re-reads its own source, strips comments and
literals, takes the span of one reference namespace and counts every
`scope::name` in it. W8a adds a FIFTH namespace, `ref_apply`, to that instrument
rather than writing a second one — reference code the instrument does not read
is reference code whose independence nothing measures, which is the reason W7b
and W7c-2 extended it too.

`ref_apply` is a from-scratch second implementation of the one-pass planner
written only from upstream's Python. The reference is INTEGER work, so "double
precision" does not apply to it and this section does not claim it: what the
reference buys is that two independently written planners agree on the id stream
and on every span, and the LOGPROB assertions in the served suite are what carry
the numeric claim.

No performance number is claimable on any axis, on this brick as on every other
one on this row.

#### 4.18.7 The plumbing below the seam carried TWO features on the first try

Recorded because #2860 asked for a `NEEDS_DECISION` if it did not, and because a
"generic in shape" claim that nobody executed is worth nothing.

It did. The only product change in this brick is the new
`src/vllm/multimodal/processing/processor.cpp` and the seam that calls it; the
scheduler, the encoder loop, the embedding gather, the prefix-cache key builder
and the model's masked scatter are untouched, and the first build with the seam
wired ran `test_openai_api_server_dots3_mm_forward` green at 28 / 16467. No
`NEEDS_DECISION` was raised and W8a was not split.

#### 4.18.8 Mutations

Each is applied to a scratch copy, built, run RED, and the tree restored
byte-for-byte with both binaries re-hashed. Numbers are in §4.18.9.

| ID | Mutation | Must red |
|---|---|---|
| M1 | keep only the FIRST feature in the seam's span loop | the three W8a served cases |
| M2 | restore the sequential two-pass expansion (each rule applied on its own, against the ORIGINAL ids) | the mixed served case |
| M3 | delete the entry-point route, so `MakeDots3NoteChatSeam` is never reached | every served case in the suite |
| M4 | make `MakeTokenTripleReplacement` key on the PAD id alone instead of the target triple, in its COHERENT form: `target = {pad_id}` **together with** `full = n pads` and `embed_offset = 0`, so the rule still describes one self-consistent replacement | the unit case that types a bare pad id into the text |
| M5 | emit spans per MODALITY instead of in stream order, in the CHAT SEAM's span loop | the mixed served case's span assertions |

Two of those rows name a PLACEMENT or a FORM, and both do so because the row
does not reproduce without it. M4's half-form — retarget to `{pad_id}` while
LEAVING the `[start] + pads + [end]` content — is a different and far more
destructive mutation, because the replacement then no longer describes what it
replaces; it is not what the recorded row measured. M5's placement is the
subject of §4.18.9's correction below.

#### 4.18.9 Evidence

Measured on this worktree at `f2930e918` + this branch, CPU only, in a
`/dev/shm` build tree configured `-DVLLM_CPP_SERVER=ON
-DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`.

**Oracle identity.** `~/_git/vllm` `git rev-parse HEAD` =
`5559679229bc961848b121ccdeaa8fa5d79bec98`, the parity pin, whose tree contains
no `dots3_note` path at all (`git ls-tree -r --name-only 5559679229 | grep -i
dots` returns `dots_ocr.py` and `dotsocr.py` only). Every anchor in this section
names `9035151d6` =
`9035151d6c9fb726181469f9e6aa9ccbf9a5dacb`, where
`vllm/models/dots3_note/common/processor.py` and
`vllm/multimodal/processing/processor.py` both exist.

**RED-before, on the tree with the three cases inverted and no implementation
present.** `test_openai_api_server_dots3_mm_forward`
`907994b6a24425ec44680763623cc78cd5635ad63b6ea6aaa2b7b33035f834a9`, **28 cases,
25 passed, 3 failed / 16377 assertions, 3 failed**. The three are exactly the
inversions, and each names the refusal that had to go:

```text
TEST CASE:  dots3-note W8a: TWO images in one request are both served, ...
  FATAL ERROR: REQUIRE( r.status == 200 ) is NOT correct!
  values: REQUIRE( 400 == 200 )
  logged: body: {"error":{"code":400,"message":"At most 1 image(s) may be provided in one prompt.",...}}

TEST CASE:  dots3-note W8a: TWO audio parts in one request are both served, ...
  FATAL ERROR: REQUIRE( r.status == 200 ) is NOT correct!
  values: REQUIRE( 400 == 200 )
  logged: body: {"error":{"code":400,"message":"At most 1 audio(s) may be provided in one prompt.",...}}

TEST CASE:  dots3-note W8a: an image and an audio part in ONE request are BOTH served, ...
  FATAL ERROR: REQUIRE( r.status == 200 ) is NOT correct!
  values: REQUIRE( 400 == 200 )
  logged: body: {"error":{"code":400,"message":"dots3-note multimodal chat seam: this request
    carries BOTH an image and an audio part. ... that is owed to W8. ..."}}
```

`test_dots3_note_audio` at that point was untouched and green at **24 / 4014**.

**GREEN-after.** `test_dots3_note_audio`
`ea48d56ce241f5b2209a8fe61bbc676b4eaa4bd624ed45079e9f292800aae816` **28 / 4206**,
`test_openai_api_server_dots3_mm_forward`
`751f6c9752fb1f7efea3088e17111e93581750c125b475c30539ba118eb1a4bb` **28 / 16467**.
Measured on the mixed request: `spans: audio [1, 8) image [11, 15)`,
`mixed vs audio-only: 0.461567, mixed vs image-only: 0.83363`. The second image
moves the first token's logprobs by up to `0.172768` and the second waveform by
up to `0.668705`.

**Mutations.** Every one built (`BUILD_RC=0` recorded for each, because a build
failure reads as a passing test), and the tree was restored with `git checkout
-- .` and rebuilt after each; the final rebuild reproduced BOTH green hashes
byte-for-byte, which is the restoration proof.

| ID | audio suite | mm-forward suite | binary sha (mm-forward) |
|---|---|---|---|
| green | 28 / 4206 | 28 / 16467 | `751f6c97…` |
| M1 | 28 / 4206 | **25 / 16397, 3 cases failed** | `d7137d8d…` |
| M2 | 28 / 4206 | **27 / 16423, 1 case failed** | `1fd5af9c…` |
| M3 | 28 / 4206 | **2 / 16353, 26 cases and 58 assertions failed** | `6469a002…` |
| M4 | **27 / 4205, 1 case failed** | 28 / 16467 | `58b9d389…` |
| M5 (as first written) | 28 / 4206 | 28 / 16467 | `71f1fc9c…` |
| M5b | 28 / 4206 | **27 / 16457, 10 assertions failed** | `4c407724…` |

Three of those rows say something the table alone does not.

**M2 is engine-FATAL, not a logprob drift.** The chained expansion puts the
image span inside the audio span, so the runner gathers 11 encoder rows for 8
masked positions and `EmbedMmDots3NoteForCausalLM`'s own balance check throws
inside the busy loop:

```text
engine-fatal: EngineCore busy loop threw: vt: Dots3NoteForCausalLM embed:
  11 gathered encoder rows for 8 masked placeholder positions. A masked scatter
  that does not balance splices vision features onto text rows.
  at src/vllm/model_executor/models/dots3_note_registry.cpp:423
api-server: 500 endpoint=/v1/chat/completions ...
```

That is a stronger red than the one #2860 predicted, and it also shows the
single-modality cases survive M2 (27 of 28 pass): chaining is only wrong when
more than ONE rule runs, which is exactly §4.18.1's claim.

**The M5 correction below and the MN1 paragraph at the end carry a SECOND
measurement, taken while repairing this slice's fresh review and not by the wave
itself.** It ran in its own `/dev/shm` build tree configured
`-DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_SERVER=OFF
-DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`, which is why its green
`test_dots3_note_audio` hashes `54fc6dc7…` where the wave's, built with the
server on, hashes `ea48d56c…`. The CASE and ASSERTION counts agree on both — 28
/ 4206 — and that is the axis the mutations are read on; the hashes are there
only to prove that each mutated binary really differed and that the tree came
back. The served suite needs the server and was not rebuilt for these two, so
every served-suite number in this section stays the wave's.

**M5 AS FIRST WRITTEN DID NOT RED, and the inertness is a property of WHERE it
was placed, not of the fixture data.** It sorted the spans ASCENDING by modality
name inside the CHAT SEAM's span loop. The reason first recorded here — that
`"audio" < "image"` is already the stream order on every case in the suite, so
the sort was a no-op — **is false. The fresh review of this slice found it
false, and the numbers below are the repair's own re-measurement.** Place the
identical ascending sort one level down, on the `applied` vector inside
`ApplyPromptReplacements` itself, and the applier suite goes RED:
`test_dots3_note_audio` **27 / 4186 passed, 1 case and 20 assertions failed**,
against the green `28 / 4206`, with the binary changed (`2d6a1709…` against the
green `54fc6dc7…`) and restored to `54fc6dc7…` byte for byte afterwards. The 20
split across the two subcases the old sentence overlooks, both of which put a
modality out of stream order on purpose:

| Subcase | assertions | failed | what it reads |
|---|---|---|---|
| "image FIRST, so the rule order and the stream order disagree" | 26 | **7** | the image span is at offset 1 and the audio span at 7, so the sort inverts them: `CHECK( 7 == 1 )` on the first offset, and `CHECK( 14 <= 1 )` on the ascending-and-disjoint check |
| "two images and two audios, INTERLEAVED" | 51 | **13** | the same inversion, and it crosses the item indices too: `CHECK( item_index 1 == 0 )`, plus `CHECK( 30 <= 1 )` |

Both subcases call `ApplyPromptReplacements` DIRECTLY, so the seam-level loop M5
mutated is downstream of them and neither can reach it. What M5 as first written
measured was therefore its own placement and not the suite's coverage, and the
distinction is the whole point of recording an inert mutation: a mutation that
never applies reads exactly like a mutation the tests survived, and the reason
given for the reading is what the next reader uses to decide whether a similar
mutation is worth running. A reason that says "the data cannot discriminate"
retires the question; the true reason — "this instrument was placed downstream
of the cases that discriminate" — does not. M5b sorts DESCENDING at the seam and
reorders for real.

**M5b reds only the DIRECT span assertions; the served request still answers
200.** At two items in one step the window `[0, 16)` covers the whole prompt, so
`GetMmFeaturesInWindow`'s two binary searches over the unsorted pair still
return `[0, 2)` and the runner recovers. The ordering requirement is therefore
LATENT at this size and the only thing gating it is the explicit
`mm_features[0].offset < mm_features[1].offset` assertion in the mixed case.
Said plainly because the alternative is a reader assuming the served path
proves it: it does not, and a chunked-prefill step or a third item is where it
would start to.

**ONE PORTED GUARANTEE IN THIS SLICE IS UNMEASURED, AND IT IS THE TIE-BREAK.**
`processor.h`'s "a tie goes to the rule that appears EARLIER in `updates`" is a
faithful port — upstream keys the choice on `(match, _next_priority(queue))`
(`processing/processor.py:871-874` @ `9035151d6`) — but nothing here detects its
inversion. MN1, from the fresh review of this slice and re-run by the repair:
turn `at < best_start` into `at <= best_start` at
`src/vllm/multimodal/processing/processor.cpp:101`, so the LAST rule wins a tie
instead of the first. The build succeeds, the binary changes (`29f9d378…`
against the green `54fc6dc7…`, restored byte for byte after), and
`test_dots3_note_audio` stays **28 / 4206 fully green**; the review measured the
served suite green under it too. **No gate is manufactured for it, because the
case is unreachable rather than untested**: dots3-note's two rules have DISTINCT
`[start, pad, end]` targets, and one id cannot be both start ids, so
`at == best_start` cannot occur at this modality set. The independence reference
cannot close it either — it shares the tie-break polarity by construction, which
is the shared-helper failure mode this row keeps naming. What is recorded is the
obligation this transfers: a THIRD modality whose target shares a first id with
another's, or any rule set with two rules on the same target, makes the tie
reachable and would otherwise inherit an unmeasured guarantee. The change that
adds one owes the first case that can reach the tie.

---

### 4.19 W9a makes the GGUF refusal REACHABLE, and stops it saying something false

**Issue: [#2882](https://github.com/mudler/vllm.cpp/issues/2882). Brick: W9a,
the refusal half of W9.** This slice ships no arithmetic and lifts no arm. It
repairs two defects in one place: a refusal a real artifact could never print,
and a sentence inside it that is not true.

#### 4.19.1 The careful refusal was unreachable, and the message that fired named nothing

`LoadDots3NoteForCausalLM` has refused `ModelSource::Kind::kGguf` by name since
W1, naming the row, the brick and the spec. **No real `dots3note` file has ever
reached it.** The entrypoint resolves a GGUF's config BEFORE it builds a
`ModelSource`: `src/vllm/entrypoints/model_loader.cpp::HfConfigFromGgufDispatch`
is called first and `src/vllm/entrypoints/model_loader.cpp::FromGguf` second
(lines 2668 and 3069 at this brick's parent `7b8b480b1`, and they move whenever
that file does, which is why the citation names the symbols).
`dots3note` is not in `kGgufArchArms`, so the dispatch fell through to
its explicit default and the file died with

```
GGUF architecture 'dots3note' is not supported by this build. GGUF
architectures supported by this build: deepseek4, muse-glimmer, qwen35, ...
```

which names neither this model, nor the row, nor the brick, nor what is owed.
The operator learns that some list does not contain their file, and nothing
about who owes the arm.

That is the #809 failure one step further out. #809 was a refusal naming the
WRONG model; this is a refusal naming NO model on a file this project knows by
name. The in-tree answer already exists and this slice copies it exactly:
`src/vllm/entrypoints/model_loader.cpp::HfConfigFromGgufDispatch` carries a
documented block of KNOWN architectures whose GGUF arm is OWED rather than
absent, and
`src/vllm/model_executor/models/nemotron_h_registry.cpp::IsNemotronHGguf` /
`src/vllm/model_executor/models/nemotron_h_registry.cpp::NemotronHGgufRefusal`,
declared in `src/vllm/model_executor/models/nemotron_h.h::NemotronHGgufRefusal`,
is its one existing member. W9a adds the second: `IsDots3NoteGguf` /
`Dots3NoteGgufRefusal`, defined in `dots3_note_registry.cpp` beside the factory
guard that throws the SAME string, so the refusal has ONE owner and both doors a
GGUF can arrive at print it.

#### 4.19.2 The refusal said llama.cpp has no such architecture, and llama.cpp merged one

The W1 text asserted, in a comment AND in the message an operator reads, that
llama.cpp has no `dots3_note` architecture and therefore no converter to reuse.
**That is false, and it was false before this slice was written.** Verified in a
local `ggml-org/llama.cpp` clone (`/home/mudler/_git/llama.cpp`, remote `origin`
= `https://github.com/ggml-org/llama.cpp`) at `origin/master` = `0ef4d560e`,
2026-09-04:

| Claim | How it was checked | Result |
|---|---|---|
| `LLM_ARCH_DOTS3NOTE -> "dots3note"` exists | `git show origin/master:src/llama-arch.cpp \| sed -n '114p'` | `{ LLM_ARCH_DOTS3NOTE, "dots3note" },` |
| PR #27060 merged | `git log -1 5a32f7b66ef6cfb3e60deea26e3454cc6ad3438c` | `model: add dots3-note (#27060)`, 2026-08-21 19:52:34 +0200 |
| ...and is on master | `git merge-base --is-ancestor 5a32f7b6… origin/master` | yes |
| PR #27524 merged | `git log -1 54ee5ee643f29abba6852903ddfdb688c2361b5b` | `mtmd: support dots3-note vision+audio (#27524)`, 2026-08-22 10:35:50 +0200 |
| ...and is on master | `git merge-base --is-ancestor 54ee5ee6… origin/master` | yes |
| a converter exists | `git show --stat 5a32f7b6…` | `conversion/dots3.py`, 195 lines, new file |
| the mtmd half exists | `git show --stat 54ee5ee6…` | `tools/mtmd/models/dots3note.cpp`, 61 lines, new file |

`5a32f7b6…` is 20 files, +1412/-9; `54ee5ee6…` is 15 files, +535/-11. Both are
squash commits with a single parent, which is why neither reads as a merge.

**The correction has to land in the MESSAGE, not only in the comment.**
`dots3_note_registry.cpp:124-125` is product output: it is the sentence an
operator gets on stderr. A record correction that repairs the surrounding
comment and leaves the false sentence in the throw is not a fix, and this
repository has the failure on file. So the new text says what is true — the arm
is owed to W9, llama.cpp DOES define the architecture, and the merge commits are
named so a reader can check rather than trust.

**What the new text deliberately does NOT say.** It does not claim this build
can serve such a file, or that the converter is reusable as-is. Whether our
loader can read llama.cpp's tensor layout is W9b/W9c's question, and #2882
records one known delta already: llama.cpp splits our fused `kv_b_proj` into
`attn_k_b` / `attn_v_b`. A refusal that oversells the position is the same class
of defect as one that undersells it.

#### 4.19.3 What the refusal now says

```
Model architecture Dots3NoteForCausalLM does not support GGUF weights yet: the
GGUF k-quant arm is OWED to W9 -- both a loader arm and, for an artifact that
needs one, a converter. Row MODEL-MM-dots3-note-dots3-note-for-causal-lm, spec
.agents/specs/dots3-note.md section 4.19. llama.cpp DOES define this
architecture -- LLM_ARCH_DOTS3NOTE -> "dots3note", ggml-org/llama.cpp
src/llama-arch.cpp, merged as 5a32f7b66ef6cfb3e60deea26e3454cc6ad3438c
("model: add dots3-note", 2026-08-21) and 54ee5ee643f29abba6852903ddfdb688c2361b5b
("mtmd: support dots3-note vision+audio", 2026-08-22) -- and published dots3note
GGUF artifacts exist, so a file reaching this refusal is a real one. This build
cannot read it yet.
```

#### 4.19.4 The gate is REACHABILITY and TEXT, not a number

§6.4 option B, and this slice is the clearest case of it on the row: it adds no
arithmetic at all, so there is nothing for an oracle to be an oracle OF. **Its
gate is that the refusal fires on the production path, and that the text it
prints is true.** Nothing here is a correctness gate in the numeric sense, and
nothing in this section should be read as one.

The smallest failing test drives a synthetic GGUF whose only content is
`general.architecture = "dots3note"` through `LoadedEngine::FromModelDir` — the
entry point every server and CLI `.gguf` argument takes — and asserts the thrown
message names dots3-note, the row and the brick, and does NOT carry the
build-level "is not supported by this build" text. It lives beside the #809
cases in `tests/vllm/test_model_loader_gguf.cpp`, which is where the dispatch's
other refusals are already held.

#### 4.19.5 What is still owed, BY NAME

Unchanged by this slice, and refused rather than deferred silently:

- **W9b — the GGUF loader arm.** Reading a `dots3note` file's tensors into
  `Dots3NoteWeights`, including llama.cpp's `attn_k_b`/`attn_v_b` split against
  our fused `kv_b_proj`.
- **W9c — the header manifest**, i.e. what `Dots3NoteHfConfigFromGguf` would
  have to read to build an `HfConfig` this row's parser accepts.
- **W9e — the mmproj arm** for the vision and audio towers.
- **W9f — the end-to-end run and the `llama-cpp` oracle pin advance.** The
  pinned oracle is `b10451` ([oracles/llama-cpp.md](../oracles/llama-cpp.md)),
  which predates both merges above, so a quant-matched comparison needs a pin
  advance and that is a row of its own.
- **The blockwise-FP8 language and audio arms** stay refused, unchanged.
- **The vision FP8 divergence** is [#2881](https://github.com/mudler/vllm.cpp/issues/2881),
  which this slice does not touch.

#### 4.19.6 #699 is UNREADABLE, not gone -- and one pointer to it was SUBSTITUTED

`gh api repos/mudler/vllm.cpp/issues/699` returns HTTP 404 while the account
reads healthy (`gh api user` succeeds, and every other issue this slice cites
reads fine). This repository has a recorded incident in which API 404s were read
as deletions and the falsehood was written into `AGENTS.md`, so this slice does
NOT conclude that #699 is gone.

**It is demonstrably still present, and the READ PATH is what is broken.** The
first draft of this section said "404, cause unknown", which is weaker than what
can be measured. Measured 2026-09-04:

| Read | 698 | 699 | 700 | 701 | 702 |
|---|---|---|---|---|---|
| `gh api .../issues/<n>` | 404 | 404 | 404 | 404 | 404 |
| `gh api .../issues/<n>/timeline` | 3 events | 13 events | 3 events | 3 events | 2 events |

699's earliest timeline event is a `referenced` at `2026-08-14T14:54:07Z`. An
issue that was deleted has no timeline to serve, and a CONTIGUOUS five-number
block cannot all have been deleted while all five answer on a second endpoint.
So the finding is "present, unreadable through this endpoint", not "deleted" --
which is the same refusal to act, held up by evidence instead of by caution.

**One pointer to #699 did leave `src/`, and it is a SUBSTITUTION.** W1's throw
ended `See .agents/specs/dots3-note.md and issue #699.`; `Dots3NoteGgufRefusal`
ends with the row ID `MODEL-MM-dots3-note-dots3-note-for-causal-lm` and `spec
.agents/specs/dots3-note.md section 4.19` instead. Measured on this branch:
`git grep -c 699 -- src/` lists 30 files at the base `3e246b34f` and 29 at the
reviewed head `a7d9c0d1f`; the one that leaves is `dots3_note_registry.cpp`
(1 line), and every other count is byte-identical, `dots3_note.h`'s 6 included.
Both trees are named because a count with no tree beside it is the shape #2323
refuses.

This spec still carries the reference, here and in many other sections, and this
paragraph deliberately does NOT quote how many. Its first draft did -- it said
31 -- and that number was already wrong by the time the section was rewritten to
discuss #699 at length. A count of one file stored as prose in that same file is
a drift lock, which is the shape AGENTS.md `## Records` names, and it is also the
"measured at the parent, quoted as if it described the head" trap. The reader who
wants the number runs `git grep -c 699 -- .agents/specs/dots3-note.md`. What is
stated here instead is the direction, which is UP, and the one line that left
`src/`, which is named above.

The reason is NOT the 404, and this section said the opposite until it was
reviewed. The refusal is product output on a user's stderr: a row ID and a spec
section resolve inside the checkout the user already has, while an issue number
resolves only through the endpoint that is currently answering 404. Substituting
the pointer that always resolves for the one that does not is the better refusal.
Deleting the RECORD would not be, and no record was deleted.

#### 4.19.7 Evidence, measured 2026-09-04

Built in `/dev/shm` with `TMPDIR` inside the build directory, `-j 2`:
`-DVLLM_CPP_SERVER=ON -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF
-DCMAKE_BUILD_TYPE=Release`. No GPU, and none is applicable: this slice runs no
arithmetic.

**Which TREE each number was measured on**, because the first draft of this
header named only `7b8b480b1` while the table below quoted the merge commit, and
an evidence table that does not name its tree is the shape #2323 names. The
RED/GREEN pair and M1/M2 were measured on `7b8b480b1` plus this branch; the fresh
review re-ran every one of them on the merge commit `a7d9c0d1f` and each number
holds. §4.19.8's numbers are a THIRD tree, the repair head, and are labelled
there rather than folded in here.

**RED before, verbatim.** `test_model_loader_gguf` at the test-only commit,
binary sha256 `736df440c58e3bfe8cfd8c314410aa9c7b4c1f519defcc37a036f23b0a57694c`:

```
[doctest] test cases: 11 |  9 passed | 2 failed | 0 skipped
[doctest] assertions: 46 | 37 passed | 9 failed |
[doctest] Status: FAILURE!
```

Nine failures, and one of them says WHY rather than merely that:

```
tests/vllm/test_model_loader_gguf.cpp:272: ERROR:
  CHECK( message.find("is not supported by this build") == std::string::npos )
  is NOT correct!
  values: CHECK( 30 == 18446744073709551615 )
```

Offset 30 is exactly past `GGUF architecture 'dots3note' `, so the message an
operator got was the build-level default and nothing else. The other eight are
`npos` on every claim the case makes about the text.

**GREEN after**, binary sha256
`edc9e007e60eeb3cc80fa78cfbd23c6a5df6af3b3691434766ce33313f62ad91`:

```
[doctest] test cases: 11 | 11 passed | 0 failed | 0 skipped
[doctest] assertions: 46 | 46 passed | 0 failed |
```

**The mutations.** Each was applied to the GREEN tree, rebuilt, run, and
restored — and the restore is proved by the binary, not asserted: rebuilding
after the last restore reproduced `edc9e007…` exactly.

| # | Mutation | Binary sha256 | Result |
|---|---|---|---|
| M1 | Delete the whole `if (vllm::IsDots3NoteGguf(gguf))` branch from `src/vllm/entrypoints/model_loader.cpp::HfConfigFromGgufDispatch` — the production call site | `336a2a74…` | **RED** 11 / 9 passed / 2 failed, 46 / 37 / 9 |
| M2 | `src/vllm/model_executor/models/dots3_note_registry.cpp::IsDots3NoteGguf` returns `false` unconditionally | `e54cefec…` | **RED** 11 / 9 passed / 2 failed, 46 / 37 / 9 |
| — | restore | `edc9e007…` | **GREEN** 11 / 11, 46 / 46 |

Both build cleanly (`ninja` exit 0), which is what makes their reds a
measurement rather than the compile failure this row has twice read as a pass.
M1 is the reachability mutation AGENTS.md's `## Nothing lands dead` asks for:
with the call site gone, nothing in the tree produces the new string, and the
suite goes red at the same nine assertions the pre-implementation RED failed on
— the same count from the opposite direction, because the file falls back to the
build-level default in both.

**What these mutations do NOT prove.** They prove the dispatch branch is reached
and that the predicate decides it. They say nothing about whether the SENTENCES
are true; that is §4.19.2's table, whose evidence is a llama.cpp checkout and
not this build. The two SHAs asserted in the test are the seam between them: the
test holds the text stable, and only a reader with the clone can check that the
text is right. A test cannot verify a claim about another repository, and this
section does not pretend otherwise.

**One sibling case had to MOVE, and it is the honest half of this slice.**
`test_dots3_note_scaffold`'s "GGUF k-quants are OWED (W9)" subcase asserted the
old literal and went red the moment the text moved into one owner. It now
asserts the new substring AND that the factory guard's bytes equal
`Dots3NoteGgufRefusal()` exactly; its own comment carried the false llama.cpp
claim and is corrected with the reason. That is a gate finding, not a bystander:
a suite that stayed green through a rewritten refusal would have been measuring
nothing about the text.

**This paragraph used to end "so the two doors cannot drift", and that was one
door.** The byte-equality above holds the FACTORY guard — the door this very
section proves a real artifact never reaches. The reachable door, the entrypoint
dispatch, was held by substrings only, and §4.19.8 measures what that costs. The
sentence is corrected rather than deleted, because the correction is the finding.

| Suite | Result |
|---|---|
| `test_model_loader_gguf` | 11 / 11 cases, 46 / 46 assertions at `a7d9c0d1f`; **13 / 13 and 57 / 57** after §4.19.8 adds two cases |
| `test_dots3_note_scaffold` | 26 / 26 cases, 110836 / 110836 assertions (110835 before, +1 for the byte-equality assertion) |
| `test_dots3_note_attn` | 51 / 51 cases, 6888 / 6888 |
| `test_dots3_note_vision` | 13 / 13 cases, 21343 / 21343 |
| `test_dots3_note_audio` | 28 / 28 cases, 4206 / 4206 |
| `test_nemotron_h_scaffold` | 14 / 14 cases, 38311 / 38311 |
| `test_model_resolver` | 11 / 11 cases, 61 / 61 |
| `test_gguf` | 36 / 36 cases, 133 / 133 |

`test_nemotron_h_scaffold` is in that list on purpose: this slice adds a second
member to the block its refusal already occupies, and the way to get that wrong
is to shadow the first one.

#### 4.19.8 The fresh review found the slice OVERSTATING ITSELF, four times

The refusal's TEXT reviewed clean: every llama.cpp claim and the artifact claim
were re-derived independently and hold. All four findings were about what this
slice claimed ABOUT ITSELF — which is the same defect class §4.19 exists to fix,
one level up. They are recorded here rather than repaired quietly, because a
correction with no record is how the next reader repeats it.

**F1 — two NEW line anchors were stale at this slice's OWN head.**
`dots3_note.h` and `tests/vllm/test_model_loader_gguf.cpp` both cited
`model_loader.cpp:2668` and `:3069`. This slice inserts four lines at `:1289`,
ABOVE both, so at the merge commit the call sites are 2672 and 3073; 2668 is a
Qwen4-Exp `VT_LOAD_STATS` comment and 3069 a platform-resolution comment, and a
reader landed in an unrelated row's prose. That is #1143 committed by the very
change that converted the OLD anchors to symbol form. Both now cite
`src/vllm/entrypoints/model_loader.cpp::HfConfigFromGgufDispatch` and
`src/vllm/entrypoints/model_loader.cpp::FromModelDir`.
`scripts/check-symbol-anchors.py` staying green was never evidence: its own
docstring says it does not verify LINE citations, so it structurally could not
see these.

**F2 — "the two doors cannot drift" was gated on the WRONG door**, and this is
the finding with teeth. The byte-equality lives in `test_dots3_note_scaffold` and
compares the FACTORY guard with `Dots3NoteGgufRefusal()` — the door §4.19.1
proves a real artifact never reaches. The REACHABLE door, the entrypoint
dispatch, was held by substrings only, and a substring set cannot see a message
that still contains every substring. `test_model_loader_gguf` now carries
`RefusalFor(GgufWithArchitecture("dots3note")) == vllm::Dots3NoteGgufRefusal()`.

**F3 — the predicate was proved to FIRE, not proved to fire ONLY for
`dots3note`.** Gross over-match was already caught (returning `true`
unconditionally reds the `mamba-unported` case), but nothing fed a NEAR MISS, so
loosening `==` to a prefix match left both suites green. Two architectures close
it: `dots3note-moe`, a plausible sibling whose GGUF is not this model's, and
`dots3_note`, the underscore spelling W1's own text used. Both must reach the
build-level default, and neither may be claimed by a row that does not owe it.

**F4 — §4.19.6's "does not remove any reference to #699" was FALSE as written**,
and the diff said so. §4.19.6 now records what actually happened and why the
substitution is the better refusal, together with the timeline evidence that
#699 is unreadable rather than gone. `squash_merge_commit_message = PR_BODY`
would have made the false sentence a permanent commit message.

**Evidence, measured 2026-09-04 on the repair head.** Same recipe as §4.19.7,
built in `/dev/shm` with `TMPDIR` inside the build directory, `-j 2`,
`-DVLLM_CPP_SERVER=ON -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF
-DCMAKE_BUILD_TYPE=Release`. Every `ninja` in the table exited 0, which is what
makes a red a measurement rather than the compile failure this row has twice
read as a pass. Mutation A is `+ " (dots3note)"` appended to what
`src/vllm/entrypoints/model_loader.cpp::HfConfigFromGgufDispatch` throws.
Mutation B loosens
`src/vllm/model_executor/models/dots3_note_registry.cpp::IsDots3NoteGguf` from
`==` to `rfind(kDots3NoteGgufArch, 0) == 0`.

| Tree | `test_model_loader_gguf` sha256 | Result | `test_dots3_note_scaffold` sha256 | Result |
|---|---|---|---|---|
| the reviewed head `a7d9c0d1f`, unmodified | `e09fc2c6…` | GREEN 11 / 11, 46 / 46 | `f1ca54d8…` | GREEN 26 / 26, 110836 / 110836 |
| **+ mutation A, NO new cases** | `0ea9158c…` | **GREEN 11 / 11, 46 / 46** | `b2b95de9…` | **GREEN 26 / 26, 110836 / 110836** |
| + the two new cases (mutation A reverted) | `c37c544c…` | GREEN 13 / 13, 57 / 57 | `f1ca54d8…` | GREEN 26 / 26, 110836 / 110836 |
| + mutation A | `c2c2da5e…` | **RED 13 / 12 passed / 1 failed, 57 / 56 / 1** | `b2b95de9…` | GREEN 26 / 26, 110836 / 110836 |
| + mutation B | `7defeb0e…` | **RED 13 / 12 passed / 1 failed, 57 / 53 / 4** | `adec3c9b…` | GREEN 26 / 26, 110836 / 110836 |
| restore | `c37c544c…` | GREEN 13 / 13, 57 / 57 | `f1ca54d8…` | GREEN 26 / 26, 110836 / 110836 |

**Row two is the finding, not a control.** With the door a real file arrives at
printing a DIFFERENT message, the whole gate this slice shipped stayed fully
green on both suites. Row four is the same mutation against the repaired suite,
and it reds on exactly one assertion — the byte-equality — while every substring
above it still matches, which is precisely why a substring set could not do this
job. Row five's four failures are all in the `dots3note-moe` iteration
(`logged: arch := dots3note-moe`); the `dots3_note` iteration is unaffected,
because a prefix match is not the separator-normalising error and that iteration
is there for a different mutation.

`test_dots3_note_scaffold` is byte-identical (`f1ca54d8…`) at the reviewed head,
after the repair, and after the restore. The repair adds nothing to it and
removes nothing from it, and the binary says so rather than the diff. The
restores are proved the same way: both green shas reproduce exactly.

**What §4.19.8 does NOT claim.** Two doors are now each held byte-exactly against
one owner, and the predicate is held against one near miss per error class. It is
still true that no test here can verify a claim about another repository
(§4.19.7), and it is still true that a third door — one nobody has written — would
be gated by nothing. The measured statement is "the two doors that exist cannot
drift", and it is now that sentence rather than the earlier one.

---

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
| `ggml-org/dots3-note-prev-GGUF` IQ2_XXS (3 shards) | 69.24 GiB | yes | yes |
| the same repo's IQ2_S (3 shards) | 77.98 GiB | yes | yes |
| the same repo's `mmproj-dots3-note-prev-Q8_0.gguf` | 7.72 GiB (8,291,026,496 B) | yes | yes |

**The last three rows read "hypothetical (ours)" until W9a, and that was
falsified**: llama.cpp merged `dots3note` on 2026-08-21 (§4.19.2) and six GGUF
repos are published, one of them from the ggml org itself. The sizes are #2882's,
read by RANGE REQUEST off the real 5.94 MB metadata shard rather than downloaded.
The mmproj row credited a `cdanis` repo until 2026-09-04; `ggml-org/dots3-note-prev-GGUF`
ships `mmproj-dots3-note-prev-Q8_0.gguf` itself at 8,291,026,496 B (HF `paths-info`,
2026-09-04), so all three rows now name ONE repo and the table needs one pin, not two.
They are the only vehicle on this table that fits any host this project reaches,
which is why W9 is the row's only route to an end-to-end run and to a
quant-matched llama.cpp denominator on a byte-identical artifact. W9a ships none
of that: it makes the refusal REACHABLE and TRUE, and W9b/W9c/W9e/W9f still owe
the arm.

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
- **W6a — the DENSE vision tower, SERVED. LANDED** (evidence §4.11,
  [#2512](https://github.com/mudler/vllm.cpp/issues/2512)). `patch_embed` ->
  blocks 0-24 (`attn.qkv` + per-head `q_norm`/`k_norm` + 2-D vision RoPE +
  bidirectional attention, then the three-tensor SwiGLU) -> `post_trunk_norm` ->
  the `patch_merger` adapter, reached from
  `ApiServer::handle_chat_completions` through the model's `encode_mm` /
  `embed_mm` hooks and its own `REGISTER_VLLM_MM_CHAT` translation unit.
  Structurally reuses `qwen3_vl_vision`'s outline; shares no code with it,
  because the two towers agree on almost nothing below that outline (RMSNorm vs
  LayerNorm, no bias, qk-norm, a three-tensor SwiGLU, a patch-merger adapter, no
  DeepStack, no position-embedding table, no M-RoPE).
- **W6b — the pyramid MoE ViT. LANDED** (evidence §4.12,
  [#2613](https://github.com/mudler/vllm.cpp/issues/2613)). Blocks 25-41:
  `mlp.gate_weight` + `mlp.router_bias`, sigmoid scoring, the
  `capacity_factor`-derived top-2, and the 608 `moe_intermediate_size` experts,
  routed through `vt::MoeRouterTopK` at one expert group, per-expert
  `layers::UnquantizedMlpGateUpSplitMethod`, and `vt::MoeCombine` carrying the
  self-normalizing divide as a constant `routed_scale`. The RELEASED checkpoint
  no longer refuses: all 2195 `vision_encoder.*` tensors load and an image
  request against it is served. It also lifted four of W6a's five deferred
  config arms — `adapter_type = pixel_shuffle_mlp`, `post_norm = false`,
  `use_qk_norm = false`, `is_causal = true` — and left `use_bias = true`
  refused by name (§4.12.7,
  [#2616](https://github.com/mudler/vllm.cpp/issues/2616)), together with the
  softmax router and the top-k-below-2 arm
  ([#2615](https://github.com/mudler/vllm.cpp/issues/2615)). The gate carries
  the SET-equality assertion on the top-k plus the printed minimum decision
  margin, because a tolerance alone cannot see a bimodal selection flip.
- **W6c — PIL's bicubic resampler, so a real image is SERVED. LANDED**
  (evidence §4.13, [#2537](https://github.com/mudler/vllm.cpp/issues/2537)).
  `PilResizeBicubicRgb` ports `ImagingResampleInner`'s 8bpc path — the `a = -0.5`
  cubic, the `max(1, in/out)` support scaling that makes a downscale an area
  average, the `(xx + 0.5) * scale` centre, the per-output weight
  normalization, the 22-bit fixed point with its `1 << 21` rounding half and
  saturating `clip8`, and the two passes over a uint8 intermediate — and
  `ProcessImage` calls it instead of throwing. An image whose sides are not
  multiples of `patch_size * merge_size` is now answered 200 rather than 400.
  It does NOT wire `resized_size`'s `detail` / `image_details` /
  `target_height` / `target_width` arguments, which are the chat seam's request
  parsing and stay owed to W8.
- **W7 — audio tower.** The `dots` stem deltas over our Whisper encoder.
- **W8 — the MM front end, narrowed twice.** What is left is **placeholder
  expansion over MORE THAN ONE ITEM AND MORE THAN ONE MODALITY**, which is
  W8a (§4.18): upstream's list of `PromptReplacement`s, each keyed by its own
  `[start, pad, end]` target and all applied in ONE pass over the id stream
  (`vllm/multimodal/processing/processor.py:423-519`, `:944-957` @ `9035151d6`),
  plus the declared limits that expansion makes honest.
  **`<|audio_comp_*|>` LEFT: W7a discharged it** — the three ids are resolved
  from the tokenizer's added tokens by string
  (`dots3_note_processor.cpp:388-390`), which is upstream's own
  `vocab[AUDIO_START]` (`common/processor.py:757-760` @ `9035151d6`) — and
  this bullet still listed it until W8a.
  **Video decode and sampling LEFT, to [#2814](https://github.com/mudler/vllm.cpp/issues/2814)**,
  which W8a WIDENS to hold it. Upstream decodes with `torchcodec`
  (`common/video.py:189-204` @ `9035151d6`) and JPEG round-trips EVERY sampled
  frame at quality 85 (`:205-211`, applied at `:281-285` and `:331`), so a
  faithful port needs a container demuxer, an H.264/VP9/AV1 bitstream decoder
  AND a JPEG codec. This tree vendors no media library — `third_party/` is
  blake3, doctest, httplib, minja, nlohmann and vulkan — and its only decoders
  are a hand-written RIFF/WAVE walker, binary PPM P6 and raw RGB. #2814 was
  scoped to audio containers, still images and audio encode; it does not
  contain a video bitstream decoder, and that is the widening W8a owes it.
  **The `include/vllm.h` multimodal request path LEFT, to
  [#2862](https://github.com/mudler/vllm.cpp/issues/2862).** There is no
  `vllm_mm_*` symbol and the header says so twice in contract language
  (`include/vllm.h:204-216`, `:288-293`); no model sets a precedent, because
  Qwen3-VL reaches its tower only through `handle_chat_completions` too; and it
  would own `scripts/abi-capability-allowlist.txt` and
  `scripts/check-surface-coverage.py`. It must serve Qwen3-VL as well or it is
  a per-model ABI, so it is a tree-wide row and not a dots3-note brick.
- **W9 — quantized arms.** Blockwise FP8 and the owed GGUF k-quant arm +
  converter. **The vision MoE's FP32-scale FP8 formula belongs HERE, not to W6**
  (moved 2026-09-01 with W6a, [#2512](https://github.com/mudler/vllm.cpp/issues/2512)).
  The evidence that moved it is the row's own W2 census: the released bf16
  `dots-studio/dots3-note-prev` carries 37944 BF16 + 62 F32 tensors and NO scale
  tensors at all (§4.4), so there is no FP8 formula anywhere in the arm W6
  loads; `MoESwiGLUFFNFP8.process_weights_after_loading` (`vision.py:245-283` @
  `9035151d6`) CASTS bf16 experts to block-FP8 at load, which is a quantized
  path this port does not take, and the `-fp8` sibling that ships those scales
  is already refused BY NAME as W9. **R5 moved with it** (§8).
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
  [porting.md](../porting.md). **FILED AGAINST W9, not W6** (moved 2026-09-01
  with W6a, [#2512](https://github.com/mudler/vllm.cpp/issues/2512)). It was
  filed against W6 when this section was written, and the row's own W2 census
  (§4.4) contradicts that: the released bf16 checkpoint ships 37944 BF16 + 62
  F32 tensors and no scale tensor at all, so W6's arm has no activation scale to
  get wrong, while the `-fp8` sibling that does ship them is refused BY NAME as
  W9. The RISK is unchanged and is not waived; only its owner moved. The
  memory-format obligation it names still binds every brick, and W6a discharged
  its own share of it in §4.11.4.
- **R6 — no llama.cpp comparison** for the GGUF arm, so the quantized floor has
  no external reference. **FALSIFIED as written, 2026-09-04 (W9a, §4.19.2).**
  llama.cpp defines `dots3note` (`LLM_ARCH_DOTS3NOTE`, merged
  `5a32f7b66ef6cfb3e60deea26e3454cc6ad3438c` and
  `54ee5ee643f29abba6852903ddfdb688c2361b5b`), ships a converter
  (`conversion/dots3.py`) and an mtmd arm, and published artifacts exist. An
  external reference is therefore REACHABLE, and the risk narrows to a
  schedulable one: the pinned `llama-cpp` oracle is `b10451`, which predates
  both merges, so the comparison needs a pin advance (W9f). Record the gap
  rather than substituting a different model's number, and do not record it as
  a wall.
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

- **A checkpoint whose `chunk_samples` is not a whole number of `token_stride`s
  serves ONE chunk and refuses more.** W7b ([#2797](https://github.com/mudler/vllm.cpp/issues/2797),
  §4.15) landed the segment loop, so audio longer than `chunk_seconds` is served
  rather than refused. What is still owed is the arm §4.15.3 names: on such a
  config upstream's prompt-side `ceil(total / stride)` (`processor.py:771`) and
  its tower-side `sum_i ceil(seg_i / stride)` (`audio.py:129-147`) disagree, so
  a multi-chunk waveform is refused BY NAME rather than spliced. No published
  checkpoint has such a config — the released one is 960000 = 750 * 1280 — and
  upstream does not compare the two expressions at all, so serving it correctly
  needs a decision about WHICH of upstream's two numbers is the placeholder
  count. Owner: this row. Tracked in this section rather than as an issue, per
  AGENTS.md's "an issue you do not fix in the same flow has to say who owns it".
- **W7c-2 LANDED, and what is left owed here is upstream's `pyav` arm, which
  nobody can gate.** ([#2828](https://github.com/mudler/vllm.cpp/issues/2828),
  §4.17.) A PCM16 WAV at any sampling rate is now SERVED: `ResampleAudioScipy`
  converts it to `audio_config.sampling_rate` before the front end, mirroring
  `resample_audio_scipy` (`vllm/multimodal/audio.py:232-250` @ `9035151d6`),
  which is an arm of upstream's own `AudioResampler` switch and which vLLM
  already ships in production for another model
  (`vllm/model_executor/models/phi4mm.py:580`). What stays REFUSED, permanently
  and by name, is upstream's DEFAULT `pyav` arm — libswresample, which is not
  bit-identical to itself across CPU dispatch on one binary and one input
  (ffmpeg 6.1.1: 24691 of 32000 samples differ, worst 9.686e-08), whose option
  defaults come from an unpinned linked binary, and whose auto-resolved `cutoff`
  is unreadable from outside the source. That refusal does not expire when
  somebody works harder; §4.17.1 is the reason. The recorded distance between
  the two, re-measured by this slice rather than relayed and reported WITH ITS
  PROBE, because the ordering is signal-dependent: on a 0 -> 7500 Hz sweep at
  44100 -> 16000 scipy is 51.78 dB from swresample's answer, soxr 44.63 and
  torchaudio 29.02, which reproduces #2828's 51.36 / 46.59 / 26.72 to about
  2 dB. On content well below the new Nyquist all three are good and soxr wins
  by 30 dB, so the number means nothing without the signal it was measured on
  (§4.17.2).
  `Ltx2ResampleWaveform` ([#2583](https://github.com/mudler/vllm.cpp/issues/2583),
  `src/vllm/model_executor/models/ltx2_audio_vae.cpp:1151`) is deliberately NOT
  reused: it is a real polyphase resampler, and it is ~25 dB FURTHER from this
  oracle on the band-limited content a speech encoder sees, because torchaudio's
  defaults are a short Hann kernel against swr's 32-tap kaiser-9. One further
  thing is owed and is a DIVERGENCE rather than a gap: `ResampleAudioScipy`
  refuses a reduced ratio whose `max(up, down)` exceeds `kMaxPolyphaseRate`
  (100000), because the filter is `20 * max(up, down) + 1` taps and the rate is
  named by the REQUEST. Upstream has no such guard. Owner: this row.
- **Every audio CONTAINER but RIFF/WAVE PCM16 — and this one is NOT owed to
  this row.** `mp3`, `flac`, `ogg` and anything else an `input_audio.format` may
  name need a demuxer this tree does not vendor. Five surfaces already refuse
  compressed media for the same missing brick, and
  [#2814](https://github.com/mudler/vllm.cpp/issues/2814) tracks it as shared
  work. W7c-1 narrowed this row's container message to say so, so a reader is
  no longer told that a `.mp3` is waiting on a dots3-note brick. Owner: #2814.
- **The `include/vllm.h` MULTIMODAL REQUEST PATH — and this one is NOT owed to
  this row either.** W8a made ONE dots3-note request carry two `mm_features`,
  and it made none of that reachable from the C ABI. There is no `vllm_mm_*`
  symbol, and the header says so twice in contract language
  (`include/vllm.h:204-216`, `:288-293`): an `image_url` or `input_audio` part
  handed to `vllm_chat` is dropped and the request is answered as text. It left
  this row because it cannot be a dots3-note brick — `Qwen3VLForConditionalGeneration`
  reaches its tower through `handle_chat_completions` too, so a path that served
  only dots3-note would be a per-model ABI — and because the change owns two
  tree-wide surfaces this row does not: `scripts/abi-capability-allowlist.txt`
  and `scripts/check-surface-coverage.py`. It is therefore a tree-wide row of
  its own rather than a W8 slice, and W8a's pull request deliberately carries no
  closing keyword for it. Owner:
  [#2862](https://github.com/mudler/vllm.cpp/issues/2862).
- **`vt::Conv2d` has no CUDA provider, and W7a's stem composition is the
  exception that records it.** `src/vt/cpu/cpu_conv2d.cpp:111` is the only
  `RegisterOp(OpId::kConv2d, ...)` in the tree, so the shared 2-D convolution
  seam resolves no op on a CUDA queue. `dots3_note_audio.cpp` composes its three
  stride-2 Conv2d stem layers as im2col + `vt::MatmulBT` instead — the same
  composition `whisper_audio.h:33` already makes for its Conv1d stem, and for
  the same stated reason — and carries ONE EXACT TRACKED EXCEPTION naming this
  gap. The gap is a `vt` one and outlives this row. Issue
  [#2709](https://github.com/mudler/vllm.cpp/issues/2709).

- **The vision MoE's SOFTMAX router arm and its top-k-below-2 arm are refused.**
  `Dots3NoteVisionRefusal` turns away a `vision_config` whose
  `router_scoring_func` is not `"sigmoid"`, and one whose
  `min(int(capacity_factor), num_routed)` is below 2 on any routed block. Both
  are arms upstream implements (`vision.py:184-185`, and the `topk > 1` guard at
  `:196` @ `9035151d6`). The reason is §4.12.3's: on those arms upstream skips
  the weight renormalization, which leaves the self-normalizing combine's
  `aggregated_gate` denominator PER TOKEN, and no op in `include/vt/ops.h`
  expresses a per-token scale on the combine — `vt::MoeCombine`'s `routed_scale`
  is one float and `vt::MulColVecF32` broadcasts over columns. Closing it means
  extending `vt::MoeCombine` with an optional `[T]` f32 divisor on both the CPU
  and CUDA kernels, which is an edit to the op DeepSeek-V2's token-exact path
  routes through, and then deleting the refusal in the same change. **Nothing
  ships either arm**: the released `dots-studio/dots3-note-prev` and
  `DotsMoEVitConfig`'s own defaults are both sigmoid at `capacity_factor` 2. The
  in-test reference already spells upstream's literal `aggregated_gate`
  division, so it will measure the new arm unchanged. Owner: this row. Issue
  [#2615](https://github.com/mudler/vllm.cpp/issues/2615).
- **`use_bias = true` is refused for the whole vision tower.** Upstream threads
  `DotsMoEVitConfig.use_bias` (`vision.py:43` @ `9035151d6`) into the attention
  `qkv`/`proj` (`vision_attention.py:143-144`), every dense block's `fc1`/`fc2`/
  `fc3` (`vision.py:129-133`) and every routed EXPERT's (`vision.py:159`) —
  1949 extra tensors on the released geometry. It is refused rather than
  implemented for three reasons, none of them effort: nothing published sets the
  key (upstream's default is `False` and the released config agrees); the shared
  `layers::MlpGateUpMethodBase` seam has no bias operand, so lifting it means
  either extending the seam every model in the tree routes its MLP through or
  writing the two GEMMs by hand beside it; and it would land UNREACHED, since
  the only production entry point that could reach the arm is a checkpoint that
  declares it and the only such checkpoint would be a fixture written to reach
  it. The refusal names the keys, the seam and the issue, and a served request
  against such a checkpoint gets HTTP 400 with the text path still answering.
  **W7a (§4.14.4) lifted the SECOND of those three reasons and only that one**:
  `layers::UnquantizedMlpGateUpBiasMethod` now exists, and it landed REACHED
  because the released AUDIO checkpoint's `fc1 [10240, 1280]` ships a `[10240]`
  bias. The first and third reasons still stand for the VISION tower — nothing
  published sets `use_bias`, so a vision arm would still land unreached — and
  the refusal is unchanged. #2616 therefore stays OPEN, and W7a's pull request
  deliberately does not carry a closing keyword for it. What remains is loading
  the vision `qkv`/`proj`/`fc` biases and deleting the refusal, gated by a
  `use_bias` fixture served end to end.
  Owner: this row. Issue
  [#2616](https://github.com/mudler/vllm.cpp/issues/2616).
- **`resized_size`'s per-request DETAIL overrides are not wired.** Upstream's
  `Dots3NoteImageProcessor.resized_size` accepts `detail`, an
  `image_details[detail]` table of `min_pixels` / `max_pixels` /
  `target_height` / `target_width`, and explicit `target_height` /
  `target_width` arguments (`common/processor.py:97-119` @ `9035151d6`). This
  port honours the `min_pixels` / `max_pixels` pair from
  `preprocessor_config.json` and ignores the other three inputs, because
  `RouteDots3NoteImageRgb` calls `ProcessImage(rgb, height, width)` with the
  decoded size and nothing else and no released `preprocessor_config.json`
  carries an `image_details` table. Closing it is a FRONT-END change — the
  OpenAI `image_url` part's `detail` field has to reach the processor call —
  and it therefore belongs to W8 with the rest of the request parsing, not to
  the resampler W6c landed (§4.13). Nothing published selects it and the
  default `detail` resolves to the config pair the port already reads, so the
  gap is invisible to every checkpoint this row can feed. **W8a (#2860) did NOT
  take it**, and the reason is the one #2616 is refused for: nothing published
  carries an `image_details` table and the default `detail` resolves to the
  config pair the port already reads, so the code would land reachable only
  through a fixture written to reach it. Owner: this row, a later W8 slice.
  Issue [#2645](https://github.com/mudler/vllm.cpp/issues/2645).
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

- **The GGUF k-quant arm itself, split into four owed slices by W9a
  ([#2882](https://github.com/mudler/vllm.cpp/issues/2882), §4.19.5).** W9a made
  the refusal reachable and true and shipped nothing else. **W9b** owes the
  loader arm, including llama.cpp's `attn_k_b`/`attn_v_b` split against our
  fused `kv_b_proj`. **W9c** owes the header manifest a `dots3note` file's
  `HfConfig` would be built from. **W9e** owes the mmproj arm for the vision and
  audio towers. **W9f** owes the end-to-end run and the `llama-cpp` oracle pin
  advance off `b10451`, which predates both llama.cpp merges. Owner: this row,
  W9. Tracked here rather than as four issues, per AGENTS.md's "an issue you do
  not fix in the same flow has to say who owns it".

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

**W8a LANDED (#2860): ONE dots3-note request now carries TWO `mm_features`, the
first request in this repository to carry more than one.** The two sequential
expanders are replaced by upstream's one-pass shape — a list of
`PromptReplacement`s each keyed by its own `[start, pad, end]` target, applied
in a single walk (`vllm/multimodal/processing/processor.py:423-519`,
`:944-957` @ `9035151d6`) — the declared ceiling rises to upstream's
`{"image": 512, "audio": 128}` with `video` left ABSENT so its refusal does not
move, and a request mixing an image and an audio part is SERVED where it was
refused by name. Everything below the seam was N-generic in shape and had never
been exercised; it carried two on the first try, with no engine change (§4.18.7).
The row stays `SPIKE`, for the reason it has always stayed `SPIKE`: the tower
serves, and the 280B-A16B model still does not fit any host this project
reaches. What LEFT W8 in the same change is video decode, to
[#2814](https://github.com/mudler/vllm.cpp/issues/2814) widened to hold it, and
the `include/vllm.h` multimodal request path, to
[#2862](https://github.com/mudler/vllm.cpp/issues/2862).

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
2195 vision and 430 audio tensors were named W6/W7 deferrals and no multimodal
front end existed. **W6a flipped it back, and this paragraph said otherwise
until W8a repaired it.** `supports_multimodal` is `true` today
(`dots3_note_registry.cpp:97`, the trail at `:54-92`), and the sentence that
read "the multimodal front end (W8) does not exist. W8 flips it back" was false
from the moment #2512 landed: an `image_url` request has been served end to end
since W6a and an `input_audio` one since W7a.

**W6a — LANDED, and this row can now be asked for something over HTTP.**
([#2512](https://github.com/mudler/vllm.cpp/issues/2512), evidence §4.11.) The
DENSE half of the vision tower — `patch_embed`, blocks 0-24, `post_trunk_norm`
and the `patch_merger` adapter — runs on a served `image_url` chat request
through `ApiServer::handle_chat_completions` -> the architecture-dispatched chat
seam -> `GPUModelRunner::execute_mm_encoder` -> `ModelRegistry::EmbedMm` ->
`ModelRegistry::Forward`. `supports_multimodal` went FALSE -> TRUE, which is the
second half of the true -> false -> true trail W5 predicted, and this time the
flag is backed: `kDots3NoteFactory` sets `encode_mm` and `embed_mm`, and
`Dots3NoteForCausalLM` has its own `REGISTER_VLLM_MM_CHAT` translation unit.

**Say the other half in the same breath, again.** The RELEASED checkpoint STILL
REFUSES, at its first MoE ViT block. 17 of its 42 vision blocks are MoE, so 1960
of the 2195 `vision_encoder.*` tensors are W6b's and the tower refuses BY NAME
before it loads one of them. That is W3's polarity applied to the second tower,
not a new exception. What W6a changed is that a config whose vision blocks are
all DENSE is now served end to end rather than refused, and that the seam every
future arm plugs into exists and is gated.

**The gate is a CONSISTENCY gate and nothing more** (§6.4 option B, §4.11.4):
an independent in-test double-precision reference written from `vision.py` and
`vision_attention.py` at `9035151d6`, sharing no helper with the implementation.
It establishes that two implementations agree. It does not establish that either
matches vLLM, and no performance number is claimed on any axis.

**Next dispatchable: W6b — the pyramid MoE ViT**, which is what the released
checkpoint's vision tower is waiting on, or W7 for the audio tower, or W9 for
the quantized arms. `## Owed` is unchanged except that `vt::QuantFp8Group`'s
missing `use_ue8m0` rounding is now recorded against W9 with the reason, because
upstream's blockwise-fp8 MoE routes through DeepGEMM with e8m0 scales, and that
R5 and the vision FP8 formula moved from W6 to W9 (§7, §8).

**W6c — LANDED, and the server will now take a photograph.** The last step of
the served image chain that still refused was its FIRST one.
`Dots3NoteImageProcessor::ProcessImage` computed the resized geometry and then
threw whenever it differed from the size it was handed, so an image was servable
only when both of its sides were already multiples of `patch_size * merge_size`
— 28 on the released checkpoint. `PilResizeBicubicRgb`
(`src/vllm/multimodal/pil_resize.cpp`) ports Pillow's `ImagingResampleInner`
8bpc path and the throw is gone; a non-conformant image is answered 200 with the
placeholder count the resized grid implies. Evidence is §4.13, issue
[#2537](https://github.com/mudler/vllm.cpp/issues/2537).

**Say what it is NOT, in the same breath.** It is PIL's resampler, not "a
bicubic kernel": the `max(1, in/out)` support scaling that turns a downscale
into an area average, the `(xx + 0.5) * scale` centre, the per-output weight
normalization and the 22-bit fixed point over a uint8 intermediate are each
load-bearing, and each has a RED-first mutation behind it in §4.13. It is also
not torchvision's, so it does not discharge Qwen3-VL's own deferred resize. And
it wires none of `resized_size`'s `detail` / `image_details` / `target_height` /
`target_width` arguments, which are request parsing and stay owed to W8
([#2645](https://github.com/mudler/vllm.cpp/issues/2645)).

**Next dispatchable: W7 for the audio tower, or W9 for the quantized arms.**
The vision half of this row is now complete for every arm any published
checkpoint selects, and what it still refuses BY NAME — `use_bias = true`
([#2616](https://github.com/mudler/vllm.cpp/issues/2616)), the softmax router
and the top-k-below-2 arm
([#2615](https://github.com/mudler/vllm.cpp/issues/2615)) — nothing published
sets.

**W7a — LANDED, and the server will now listen to a recording.**
([#2703](https://github.com/mudler/vllm.cpp/issues/2703), evidence §4.14.) An
OpenAI `input_audio` chat part now reaches the 32-layer `dots` speech encoder
and produces audio rows in the prompt embeddings, through the same production
chain W6a built for images: `ApiServer::handle_chat_completions` -> the
architecture-dispatched chat seam -> `GPUModelRunner::execute_mm_encoder` ->
`ModelRegistry::EmbedMm` -> `ModelRegistry::Forward`. Before it, the request
died at the entrypoint with HTTP 400 "At most 0 audio(s) may be provided in one
prompt.", because the seam's supported-limit map declared only `{"image", 1}`.

**Three things landed with it that are not the tower.** The mel filterbank is
now a SHARED seam, `vllm::multimodal::MelFilterBankSlaney`, extracted from
Parakeet's and used by both callers — and it is gated against
`voxtral_mel_filters_f32.bin`, a committed [201, 128] `float32` fixture that a
third party produced, which it reproduces BIT-FOR-BIT. That is the only place on
this row where a number is checked against something this repository did not
also write. `layers::MlpGateUpMethodBase` gained a BIAS arm, reached by the
audio `fc1 [10240, 1280]` + `[10240]`; §4.14.4 says why that does not close
[#2616](https://github.com/mudler/vllm.cpp/issues/2616). And **W7 IS AUDIO, W8
IS VIDEO** — the loader has said so since W2 and two production refusal messages
said the opposite; all four surfaces now agree.

**Say the other half in the same breath.** A waveform longer than
`chunk_seconds` is REFUSED BY NAME to W7b, and any container or sampling rate
but PCM16 mono WAV at 16 kHz is refused to W7c. Both are in `## Owed` with the
reason. The gate is a CONSISTENCY gate (§6.4 option B): TWO independent in-test
double-precision references, one for the front end and one for the tower,
sharing no helper with the implementation. No performance number is claimed on
any axis.

**W7b — LANDED, and the clip may now be a recording.**
([#2797](https://github.com/mudler/vllm.cpp/issues/2797), evidence §4.15.) The
`chunk_seconds` refusal is gone: a waveform of any length is sliced into
`chunk_seconds` segments, each padded to `chunk_samples` and mel'd on its own,
each run through the tower at its OWN valid length, and the per-segment row
slices concatenated in order (`audio.py:193-234` @ `9035151d6`). Upstream
batches the segments into one encoder call and this port loops; §4.15.2 shows
those are the same numbers, because the varlen pack gives each chunk its own
`cu_seqlens` window and restarts its rope positions at 0, so the chunks never
interact. `NumAudioTokens` is UNCHANGED — #2797 checked that upstream's
`(n-1)//1280 + 1` and W7a's `ceil(n/1280)` are the same function for every
`n >= 1` and differ only at `n == 0`, where a literal C++ transcription would
invent a token.

**A THIRD reference namespace came with it,** `ref_chunks`, for the
segmentation geometry alone, under W7a's existing run-time enumeration
instrument rather than a second one. The seams a tolerance cannot see — an
off-by-one slice, a reversed concatenation, an untruncated short chunk, a mask
taken from the padded length — are gated as row counts and boundary rows at a
geometry chosen so none of them can alias: three chunks, the last one short.

**W7c-1 — LANDED, and a multi-channel WAV is served.**
([#2813](https://github.com/mudler/vllm.cpp/issues/2813), evidence §4.16.)
`DecodeWavPcm16MeanToMono` sits beside `DecodeWavPcm16Mono` over one shared
`ParseWavPcm16` walk, so the channels are mean-reduced exactly as upstream's
`load_audio(..., mono=True)` reduces them. The mean is exact in `int32` and
bit-identical to upstream's `float32` mean for every power-of-two channel count
**up to 512**, a bound that is provable and TIGHT.

**W7c-2 — LANDED, and the WAV no longer has to be 16 kHz.**
([#2828](https://github.com/mudler/vllm.cpp/issues/2828), evidence §4.17.)
`ResampleAudioScipy` converts a PCM16 WAV at any rate to
`audio_config.sampling_rate` before the front end. It is a **recorded
divergence, and the first one on this row**: upstream's default resampler is
libswresample through PyAV, which is not bit-identical to itself across CPU
dispatch on one binary and one input, so no bit-exact gate against it can exist.
What is implemented is `resample_audio_scipy`, ANOTHER ARM OF UPSTREAM'S OWN
SWITCH, which vLLM ships in production for phi4mm. The gate is a CONSISTENCY
gate against `scipy.signal.resample_poly` at its defaults, with committed
goldens for four rate pairs and an aliasing fifth, and §4.17.7 states in its own
words what that does and does not establish. A FOURTH reference namespace,
`ref_resample`, came with it, under W7a's existing enumeration instrument rather
than a second one.

**Next dispatchable: W8 for video and the MM ABI, or W9 for the quantized
arms.**
