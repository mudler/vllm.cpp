# DeepSeek-V4-Flash-Vision-Exp

- **Row:** `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`
- **Issue:** [#2411](https://github.com/mudler/vllm.cpp/issues/2411)
- **Base:** `9fa3be3884076124cc90dd911af1c9bc08548e0e`
- **State on the spec commit:** `READY`
- **Git integration:** one pull request. This specification commit precedes every
  implementation commit in that pull request, as the developer selected on
  2026-08-31.
- **Checkpoint pin:** `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at
  `86f746b36186f0e567729a5c06a8c918caba82a9`.

## Scope

Port the image-understanding path of `DeepSeek-V4-Flash-Vision-Exp` through the
existing DeepSeek-V4 model and the shared multimodal engine. Completion means a
user can send one or more interleaved OpenAI `image_url` content blocks through
the production server or the public C ABI, load a documented quantized
checkpoint, and run ordinary autoregressive generation with image semantics
gated against the model author's pinned PyTorch runtime.

This is a separate multimodal row even though the checkpoint declares the same
`DeepseekV4ForCausalLM` architecture string as the text model. The existing text
row has its own oracle evidence, weight vehicles, lifecycle and performance
history. This row adds an input modality, a vision tower, different attention
visibility and a different runnable artifact. Folding it into the text row
would make the text capability appear incomplete while this work is in flight.

The port includes:

- the model-author prompt encoder for text, thinking, tools and interleaved image
  blocks;
- image loading, resizing, padding, normalization and patchification;
- the 32-layer vision transformer and the downsample-3 aligner;
- the learned image start, end, newline and padding embeddings;
- image-block N-layout expansion and the exact permutation from aligned image
  cells to prompt rows;
- image-span visibility inside DeepSeek-V4 attention and atomic image prefill;
- official FP8/FP4 safetensors loading and a runnable GGUF k-quant arm;
- `ModelRegistry::Forward`, runner, server and `include/vllm.h` reachability;
- text-only DeepSeek-V4 inertness, real-checkpoint correctness and speed gates.

## Non-goals

- **DSpark speculative decoding.** The checkpoint contains the DSpark tail and
  ordinary autoregressive decode does not require it. The model loader must
  account for those tensors and an explicit DSpark request must route to or
  refuse with `MODEL-SPEC-deepseek-v4-dspark-deepseek-v4-for-causal-lm` named.
  This row does not claim that optional speculator.
- Video and audio. The released model accepts images only.
- A DeepSeek-only server, multimodal input container or attention stack. Shared
  seams are extended where their current contracts are too narrow.
- ~~A sidecar format that combines a text GGUF with separate vision weights.~~
  **Withdrawn on 2026-09-05 by developer direction.** The developer named
  `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF` as the artifact this row has to
  run. That repository ships the llama.cpp two-file shape: a split text GGUF plus
  a `mmproj-BF16.gguf`. The original non-goal was written when no published
  quantization of this checkpoint existed and the row would have had to produce
  one; producing a private combined artifact nobody downloads is what would now
  be the assembly recipe. This tree already reads the two-file shape in
  `src/vllm/model_executor/models/clip_mmproj_gguf.cpp` (row `LOAD-GGUF-MMPROJ`,
  #821), so the seam exists and no new container is invented. What survives of
  the non-goal is its intent: a user names a documented repository and revision,
  and the loader finds both files itself.
- A token-only proof for the tower or attention visibility. Stage numerics and
  memory format are load-bearing because an omitted image mechanism can leave an
  argmax unchanged.

## Released artifact and geometry

The pinned Hugging Face index reports `167,811,372,792` bytes, or 156.287 GiB,
across 48 safetensors shards. It does not fit a 119 GiB GB10. The model author's
reference conversion and launch recipe use tensor parallelism 4. A single-device
production gate therefore depends on the GGUF k-quant arm; the official
safetensors arm remains a multi-device gate.

### The shipped quantized vehicle

`unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF` at revision
`b977d3c0ea2da58dbc12ddae8fb8951a7b3854d0`, read on 2026-09-05. It publishes ten
imatrix quantizations of the language model, each split across three to five
shards, and one vision file shared by all of them. The GGUF `general.architecture`
is `deepseek4`; the mmproj's is `clip` with `clip.projector_type = deepseek4v`.

`UD-IQ1_S` is the smallest complete arm and the one a single 119 GiB GB10 can
hold. Its identity, from the Hugging Face `paths-info` API at that revision:

| File | Bytes | SHA-256 |
|---|---:|---|
| `UD-IQ1_S/...-00001-of-00003.gguf` | 5,305,248 | `be862fb3ecdeb99a9a47fabd091b9c7bd32d0de89c9a85589cd007b822bb6305` |
| `UD-IQ1_S/...-00002-of-00003.gguf` | 49,991,832,128 | `c21604991c40674ac1612f16dcedf84b857bb5a2bace00b47a7ab7e5f5e3296e` |
| `UD-IQ1_S/...-00003-of-00003.gguf` | 32,441,484,736 | `8326a8a98fb224a16f8e83e6236fc346222bc9131f6495644c5c988b8a6101f4` |
| `mmproj-BF16.gguf` | 934,462,656 | `e4914c6c8063d01f4cbb6dafdf2f959c7d06fbe8ad11ae5b11ad032edd42642e` |

That is 82,438,622,112 bytes of language weights (76.78 GiB) plus 0.870 GiB of
vision weights, 77.65 GiB resident before KV cache and activations. A repository
id alone is not a pin, so the revision and every SHA-256 above are load-bearing.

The mmproj header confirms the geometry this spec derived from `config.json`:
427 tensors, `clip.vision.block_count = 32`, `embedding_length = 1024`,
`feed_forward_length = 2816`, `attention.head_count = 16`, `patch_size = 14`,
`projection_dim = 4096`, `projector.scale_factor = 3`,
`image_min_pixels = 147456`, `use_silu = true`, and **no position-embedding
tensor**, which is what makes the 2-D RoPE load-bearing rather than optional. The
aligner is `mm.1.weight [9216, 4096]` and `mm.2.weight [4096, 4096]`, so the 3x3
unfold of a 1024-wide tower is exactly `mm.1`'s input. The four learned vectors
are `v.token_embd.img_start`, `v.token_embd.img_end`, `v.token_embd.img_pad` and
`v.image_newline`, all f32 `[4096]`.

The released `config.json` resolves:

| Field | Value |
|---|---:|
| architecture / model type | `DeepseekV4ForCausalLM` / `deepseek_v4` |
| language hidden / layers | 4096 / 43 |
| vocabulary / context | 129280 / 1048576 |
| routed / active / shared experts | 256 / 6 / 1 |
| official quantization | FP8 E4M3 weights with UE8M0 scales; FP4 experts |
| vision layers / hidden / heads | 32 / 1024 / 16 |
| vision MLP width | 2816 |
| patch / downsample | 14 / 3 |
| maximum aligned image tokens | 384 |
| minimum pixels / maximum width-height ratio | 147456 / 8 |
| DSpark block / target layers / Markov rank | 5 / `[40, 41, 42]` / 256 |

From those dimensions, the vision tower, aligner and four learned image vectors
contain about 466.4 million parameters and occupy about 0.869 GiB at BF16. The
text backbone dominates residency. Keep the vision arm BF16 unless a measured
profile identifies it as the memory or throughput limiter; quantizing it by
habit would add error to the only new modality for less than one GiB saved.

## Oracle decision

### vLLM remains primary where it implements behavior

At the project parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`, vLLM
registers DeepSeek-V4 as text generation. The local port map and
`src/vllm/model_executor/models/deepseek_v4_registry.cpp` both record the same
text-only classification. That oracle still defines the text backbone, DSA,
MHC, MoE, KV-cache, sampling and serving behavior that this row reuses.

The vision variant is absent from the pin. Code search on 2026-08-31 at vLLM
main `dafbef15a1c879c64ebb99427917e4ca8d5bca1e` found no exact model identifier
and no `vision_n_layers` implementation. The only vLLM pull request returned by
the model-name search was #41834, which is SM12x support for text
DeepSeek-V4 Flash, not this vision path.

### Existing secondary oracles do not implement the whole model

- Transformers main `a3f3da8f87dc65d724d500eeb44777e4716aaa46`
  carries `models/deepseek_v4` for text, but no `vision_n_layers` path. The
  checkpoint uses Transformers for tokenizer utilities; its vision model is not
  a Transformers model implementation.
- SGLang main `52e1c24744bf4efe75fe976e26596ae1c9f279e2` and vLLM-Omni main
  `b81aeb7b86837f6fe8956f3aef83798ad26c5a26` contain no exact model
  implementation in code search on 2026-08-31.
- llama.cpp **had** no released DeepSeek-V4 Vision architecture when this spec
  was written. That changed on 2026-09-02. See the next section.

Calling the model-author runtime the `transformers` oracle would be false. The
executing vision code lives in the Hugging Face checkpoint repository under
`inference/`, not in `huggingface/transformers`.

### llama.cpp now implements this model, and is pinned separately

`ggml-org/llama.cpp` merged `#28133` (the `deepseek4v` clip projector, image
preprocessor and mmproj container) and `#28154` (the language-side vision
behaviour) on 2026-09-02. Release `b10766` is
`9400c8946e4da5e7694f2c26d6d4e50e14b690fa`, the merge commit of `#28154`, and it
is the first release that carries `tools/mtmd/models/deepseek4v.cpp`. The stock
[`llama-cpp`](../oracles/llama-cpp.md) pin `b10451` returns HTTP 404 for that
path and is 315 commits behind it.

This is registered as its own oracle, [`llama-cpp-dsv4vision`](../oracles/llama-cpp-dsv4vision.md),
rather than by advancing the stock pin, because every floor already measured
against `b10451` means "what that release does".

What it buys this row is not a second algorithm source. It is the first
**runnable** reference for the exact artifact the developer named: it converts
and loads `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF`, on hardware this project
leases, at a quantization our arm can match. The model author's TP4 runtime
remains the algorithm oracle and outranks it wherever the two disagree; vLLM
outranks both wherever vLLM implements the behaviour.

Two llama.cpp approximations are recorded in the oracle file and are NOT mirrored
without checking the model author: it selects the vision routing bias per ubatch
rather than per token, and it drops hash-layer `tid2eid` routing entirely for a
media ubatch.

### New secondary oracle

Register `deepseek-v4-vision`, pinned to the checkpoint repository revision
`86f746b36186f0e567729a5c06a8c918caba82a9`. Its scope is only the vision-variant
behavior that vLLM does not implement: prompt encoding, image preprocessing,
ViT, aligner, image embedding merge and image-span visibility. vLLM remains the
source for every shared DeepSeek-V4 behavior.

The new oracle starts with `gateable = no`. No session in this repository has
built the pinned runtime and run the pinned model. The official artifact needs
156.287 GiB plus runtime state and the author documents TP4. The pin becomes
gateable only after a leased job records all of these on the pinned revision:

1. the runtime builds or installs from its declared requirements;
2. all 48 shards pass the index completeness check in `inference/convert.py`;
3. the two-image reference prompt reaches generation on real weights;
4. a deterministic greedy run and the W1-W2 stage goldens are committed as
   evidence.

The checkpoint README names `inference/test_image_processor.py`, but that file
is absent at the pin. This is an upstream evidence gap, not permission to invent
preprocessing behavior. W1 executes `image_processor.py` directly and commits
its own golden inputs and outputs.

## Upstream chain

All rows below are from
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@86f746b36186f0e567729a5c06a8c918caba82a9`.

| Behavior | Pinned source |
|---|---|
| image placeholder | `encoding/encoding_dsv4.py:32` |
| tagged image syntax | `encoding/encoding_dsv4.py:613-637` |
| image content extraction | `encoding/encoding_dsv4.py:641-730` |
| OpenAI message encoding | `encoding/encoding_dsv4.py:733-780` |
| prompt/error tests | `encoding/test_encoding_dsv4.py:1-180` |
| resize budget and grid token count | `inference/image_processor.py:23-69` |
| bytes, URL and path loading | `inference/image_processor.py:71-126` |
| N-layout and aligner permutation | `inference/image_processor.py:128-148` |
| placeholder expansion | `inference/image_processor.py:150-176` |
| 2-D vision RoPE | `inference/vision.py:8-19` |
| ViT patch, attention, MLP and blocks | `inference/vision.py:21-94` |
| downsample-3 aligner | `inference/vision.py:96-109` |
| image visibility counts | `inference/model.py:276-287` |
| visible sliding-window indices | `inference/model.py:289-299` |
| DeepSeek attention consumption | `inference/model.py:464-540` |
| vision construction | `inference/model.py:904-947` |
| encode and merge | `inference/model.py:949-966` |
| forward and atomic-prefill guard | `inference/model.py:968-990` |
| tensor-parallel conversion | `inference/convert.py:65-150` |
| reference generation loop | `inference/generate.py:27-90` |

### Anchors at `llama-cpp-dsv4vision` (`b10766`)

Read for the GGUF container and as the runnable cross-check. They are never the
mirror source.

| Behaviour | Pinned source |
|---|---|
| mmproj tensor names and block layout helper | `tools/mtmd/clip-impl.h` (`TN_TOK_IMG_START/_END/_PAD`, `dsv4_get_block_layout`) |
| `deepseek4v` hyper-parameters from `clip.*` | `tools/mtmd/clip.cpp::clip_model_loader`, `PROJECTOR_TYPE_DEEPSEEK4V` case |
| tower, aligner and block assembly graph | `tools/mtmd/models/deepseek4v.cpp` |
| the N-layout permutation, as an index vector | `tools/mtmd/clip.cpp::clip_encode`, `PROJECTOR_TYPE_DEEPSEEK4V` case |
| resize solver and image loading | `tools/mtmd/mtmd-image.cpp::mtmd_image_preprocessor_deepseek4v` |
| image span decodes non-causally | `tools/mtmd/mtmd.cpp::mtmd_decode_use_non_causal` |
| **the vision routing bias** | `src/models/deepseek4.cpp`, `ffn_exp_probs_b_vl` |
| **SWA suppressed inside the image span** | `src/llama-hparams.h::swa_full_non_causal`, `src/llama-kv-cache.cpp::set_input_kq_mask_impl` |
| converter drops `aligner.*`, `image_*` and hash-layer `ffn.gate.bias` | `conversion/deepseek.py` |

## Language-side vision behaviour this spec originally missed

Two DeepSeek-V4 **language** behaviours change when the input carries an image.
Neither appears in the design section above, both are load-bearing, and both are
exactly the failure mode risk 3 names: dropping either one leaves an argmax
plausible and a token gate green.

### 1. `exp_probs_b_vl`, a second MoE routing bias

The shard in question belongs to the **language half of the VISION repository**,
`unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF`. Its first shard holds 43 tensors and
nothing else: one `blk.N.exp_probs_b_vl.bias`, f32 `[256]`, for every one of the
43 language layers. A genuine DeepSeek-V4 **text** checkpoint carries none of
these tensors, which is why every loader arm takes this one as optional and why
a text checkpoint stays byte-identical without it. It is the expert-probability
bias the router adds when the token being routed is an image token, in place of
the text `exp_probs_b`.

For the three hash layers (`deepseek4.hash_layer_count = 3`) it does more than
substitute a bias. Text tokens on a hash layer are routed by the `tid2eid` hash
table and take no bias at all; the converter drops `ffn.gate.bias` there for
that reason. An image token has no meaningful token id to hash, so on those
layers `exp_probs_b_vl` **replaces the hash routing itself**.

Our loader must therefore account for `exp_probs_b_vl` on all 43 layers and
select it per token, and `deepseek_v4_moe.cpp` must take the vision bias on the
image rows. `src/vllm/model_executor/models/deepseek_v4_weights.cpp` already
reads `exp_probs_b.bias` in both the hash-layer and noaux_tc arms, so this is a
scoped extension of an existing accounting path, not a new one.

**llama.cpp's version is coarser than ours may be.** It selects the vision bias
for the whole ubatch whenever `ubatch.embd != nullptr`, so a mixed text/image
ubatch routes its text rows on the vision bias too. Mirror the model author's
per-token rule, and record the divergence from llama.cpp rather than copying it.

### 2. SWA does not apply inside the image span

`deepseek4.attention.sliding_window = 128`. The pinned reference lets the tokens
of one image span attend across the whole span, and window-clips only the older
tokens outside it. llama.cpp models this as `swa_full_non_causal`: when the
batch decodes non-causally, the window mask is skipped for positions at or after
the span start, and applied normally below it.

This is the same rule as the spec's existing visible-window design
(`inference/model.py:289-299`), stated on the mask instead of on the index list.
W4 owns it, and its test must be an index or mask test: a 128-token window with a
384-token image span is a case where a token gate can pass while more than half
the span is invisible.

The model-author tests are ported with their parameters and failures. This
includes plain and multi-turn text stability, top-level image blocks, tagged
text equivalence, multiple-image order, TXT/JSON equivalence, malformed tags,
nested tool-result images, tool-role images, context-image exclusion,
user-injected placeholder refusal and missing-source refusal.

## Port map

| Pinned upstream | Local target | Decision |
|---|---|---|
| `encoding/encoding_dsv4.py` | `src/vllm/multimodal/deepseek_v4_processor.cpp` and the shared chat renderer | Port all image-bearing message forms and failures; keep the existing DeepSeek output parser |
| `inference/image_processor.py` | `include/vllm/multimodal/deepseek_v4_processor.h` and its implementation | New model processor over existing `MultiModalInputs`; no Qwen marker or one-image limit |
| `inference/vision.py` | `include/vllm/model_executor/models/deepseek_v4_vision.h` and `src/vllm/model_executor/models/deepseek_v4_vision.cpp` | New model composition over existing `vt` operations |
| `inference/model.py:276-299,464-540` | existing DeepSeek attention metadata and `dense_attn::AttnBlock` | Extend visible-window metadata; no second cache or attention stack |
| `inference/model.py:904-990` | `deepseek_v4.h`, `deepseek_v4.cpp` and `deepseek_v4_registry.cpp` | Optional tower/merge selected by config and input; text path stays unchanged |
| released safetensors index and `inference/convert.py` | `deepseek_v4_weights.cpp` | Account for every official tensor, including `exp_probs_b_vl` on all 43 layers |
| `unsloth/...-GGUF` mmproj + `deepseek4v` projector | `clip_mmproj_gguf.cpp` and `deepseek_v4_vision.cpp` | Extend the existing mmproj reader with the `deepseek4v` projector type and its four sentinel vectors; refuse every other type by name as it does today |
| `unsloth/...-GGUF` split text shards | `deepseek_v4_weights.cpp` GGUF arm | Load `blk.*.exp_probs_b_vl.bias` beside the existing `exp_probs_b.bias` |
| `inference/model.py` router bias for image tokens | `deepseek_v4_moe.cpp` | Select the vision bias per token; on hash layers it replaces `tid2eid` routing |
| `inference/model.py:289-299` window, as a mask | DeepSeek attention metadata | Suppress the 128-token window inside the image span only |
| OpenAI image content blocks | shared `chat_mm` and runner preparation | Model-selected placeholders, multiple images and production reachability |

## Our baseline

### DeepSeek-V4 text backbone

`include/vllm/model_executor/models/deepseek_v4.h` and the corresponding
`deepseek_v4*.cpp` files already carry the released 43-layer geometry, DSA
indexer/compressor, 512-wide latent attention, MHC, sqrt-softplus/hash MoE,
DSpark tensor recognition, safetensors accounting, GGUF k-quants and EXL3. This
row extends those types with optional vision fields and weights. It does not
fork a second language model.

The registry currently writes `supports_multimodal = false`. The architecture
string cannot distinguish the text and vision checkpoints. The registration
therefore advertises that the architecture *can* accept multimodal inputs, while
`vision_n_layers == 0` keeps a loaded text checkpoint byte-identical and tower
free.

### Multimodal engine

The reusable production seam is
`include/vllm/model_executor/models/model_registry.h:299-356`:
`MultiModalForwardInput::inputs_embeds` is an already-merged BF16 device tensor.
DeepSeek does not need Qwen MRoPE, DeepStack or Gemma PLE. Its registered forward
reads `inputs_embeds` when present and otherwise follows the existing token-id
embedding path.

`include/vllm/multimodal/inputs.h:20-92` already carries per-image patch rows,
grid dimensions, expanded prompt ids and `MultiModalFeatureSpec` offsets. The
DeepSeek processor can use that container without adding a competing request
type.

The server seam is too narrow today. `MakeQwen3VLImageChatFn` injects Qwen
markers, caps the request at one image and keeps only the first image pointer.
`server_main.cpp` constructs that processor whenever it sees
`preprocessor_config.json`, and its production codec refuses PNG/JPEG. W5 turns
this into model-selected shared processing and keeps the Qwen path unchanged.

### Operators

The vision tower composes existing `vt` operations: BF16 matmul with bias, full
non-causal dense attention, 2-D rotary application, RMSNorm, SiLU, GELU and
padding/reorder. New model TUs own the composition and weight layout. A new
kernel is justified only by a profile after correctness; no kernel row is
created by this spec.

## Design and data flow

1. The chat renderer injects `<｜deepseek_image｜>` at each image content block in
   source order and runs the pinned DeepSeek template behavior.
2. The tokenizer resolves one placeholder id per image. The processor rejects a
   mismatch between placeholder count and image count.
3. Each image is decoded to RGB, applies the exact aspect and minimum-pixel
   rules, is resized/padded, normalized to `[-1, 1]`, cast to BF16 and split into
   14x14 patches.
4. `build_image_block` computes start padding, start/end markers, newline rows,
   row-pair reorder and aligner permutation. It replaces the placeholder with
   sentinel ids `vocab_size + type` and records the feature offset and length.
5. The DeepSeek ViT runs full bidirectional attention with 2-D RoPE. The aligner
   pads the patch grid to a multiple of three, unfolds non-overlapping 3x3 cells,
   and projects them through GELU into width 4096.
6. The model embeds normal token ids. Sentinel ids produce zero from the sharded
   embedding lookup, then the image merge replaces every sentinel row with the
   matching learned vector or aligned image vector.
7. The merged device tensor enters the existing MHC language stack. During
   prefill, image-start and image-end ids derive left/right visibility counts.
   DeepSeek attention extends its window across the active image span. The
   entire span must be in one prefill chunk.
8. Decode receives only vocabulary ids, reuses the normal DeepSeek caches and
   never reruns the vision tower.

The processor supports multiple interleaved images because the pinned encoder
and example do. Limits come from `MultiModalConfig`; no lower hard-coded
one-image ceiling is added.

## Weight and quantization contract

### Official arm

The safetensors loader accounts for and loads every on-disk tensor. The new
families are `vision.*`, `aligner.*`, `image_start`, `image_end`,
`image_newline` and `image_pad`. It must also retain the existing complete
DeepSeek text and `mtp.*` accounting. A header-only structural gate requires
`enumerated == present` and zero unaccounted tensors on the pinned index.

The official arm retains the checkpoint's FP8 E4M3/UE8M0 linears and FP4 routed
experts. Unsupported storage variants refuse by name; they do not widen
silently. The vision tower inherits the resolved model dtype, with FP32 only for
normalization and RoPE intermediates where the pinned runtime widens them.

### GGUF arm

**The vehicle is `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF` at
`b977d3c0ea2da58dbc12ddae8fb8951a7b3854d0`, not an artifact this row produces.**
It is a published third-party imatrix quantization, it is what users download,
and `llama-cpp-dsv4vision` runs it, so it is the only quantized arm with a
runnable denominator. `docs/USAGE.md` labels it as third-party.

It has the llama.cpp two-file shape, which this tree already reads:

- the split language shards carry `general.architecture = deepseek4` and load
  through the existing `deepseek_v4_weights.cpp` GGUF arm, extended with
  `blk.*.exp_probs_b_vl.bias`;
- `mmproj-BF16.gguf` carries `general.architecture = clip`,
  `clip.projector_type = deepseek4v`, and loads through
  `clip_mmproj_gguf.cpp`, whose scope today is `qwen3vl_merger` and whose
  refusal of every other projector type is by name. `deepseek4v` is added to
  that reader; nothing else about its contract changes.

The vision weights stay BF16, which is what unsloth already ships, so the 0.870
GiB figure this spec derived is the shipped one rather than a target.

The user names one repository and revision. The loader resolves both files from
it. Requiring a user to hand-assemble two paths would be the assembly recipe the
withdrawn non-goal was written against, and is refused.

Before capability publication, `docs/USAGE.md` records the repository, revision,
every shard filename, byte size and SHA-256, the resident size, the official
48-shard arm, and every refused arm by name. Loader scaffolding is not model
support: the arm is done when this artifact generates from an image through the
production entry point.

## Dependencies

- Landed `MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm`: language weights,
  DSA, MHC, MoE, KV-cache and ordinary decode.
- Landed multimodal request and device-embedding seams:
  `MultiModalInputs`, `MultiModalFeatureSpec` and `MultiModalForwardInput`.
- The registered `deepseek-v4-vision` oracle pin. Its first real run stays owed
  by #2411 and does not become a static-source pass.
- An eligible leased TP4 topology for the official arm and one device with enough
  memory for the combined GGUF arm.
- Device execution uses the resource-controller fleet only. Every CUDA, ROCm or
  Vulkan command runs inside `rc run` after `rc describe`; no direct SSH may
  substitute for a lease. The 2026-08-31 fleet provides NVIDIA devices
  `dgx:gpu0`, `thor:gpu0` and `orin:gpu0`, plus AMD `strix:gpu0`.
- Existing `MODEL-SPEC-deepseek-v4-dspark-deepseek-v4-for-causal-lm` ownership
  for optional DSpark. This row accounts for that tail but does not absorb it.

## Shared seams

- `ModelRegistry::Forward` is the only model forward entry.
- `dense_attn::AttnBlock` and the existing DeepSeek cache topology remain the
  attention/KV surface. Image visibility extends their metadata; it does not
  create an unpaged cache.
- `vt::FusedChain` remains the fusion surface.
- Mergeable MLP projections route through `layers::MlpGateUpMethodBase` and
  `vt::MergedGemmGroup` where the current DeepSeek implementation does.
- `MultiModalForwardInput::inputs_embeds` is the model/runner boundary.
- `MultiModalInputs` and `MultiModalFeatureSpec` are the request/processor
  boundary.
- `include/vllm.h` exposes the capability. The server and examples are clients;
  they do not include internal model headers.

## Work breakdown

| Wave | Scope | Observable completion |
|---|---|---|
| W0 | This spec, issue, oracle pin and roadmap row | `READY`; oracle explicitly `gateable = no`; record gates pass |
| W1 | Prompt encoder and image processor | All pinned encoding cases ported; processor goldens cover resize boundaries, wide images, start-position padding, multi-image order and named failures |
| W2 | Vision tower and aligner | Reduced-shape and real-weight stage outputs agree with the pinned oracle within recorded numeric bounds; wrong RoPE axis, attention causality, downsample order and GELU each make the focused gate red |
| W3 | Official weights and the unsloth GGUF arm | Pinned safetensors index has zero unaccounted tensors, `exp_probs_b_vl` included; the pinned unsloth `UD-IQ1_S` shards and `mmproj-BF16.gguf` both load from one named repository and revision and fit one gate device |
| W4 | Merge, visibility and cached language forward | Registered forward consumes image embeddings, image-span attention matches the oracle, image prefill is atomic, decode does not rerun vision, text-only DeepSeek remains byte-identical |
| W5 | Runner, public ABI and OpenAI serving | Multiple data-URI and HTTP(S) PNG/JPEG images reach `ModelRegistry::Forward` in order; Qwen and Gemma multimodal smoke cases remain unchanged |
| W6 | Real-checkpoint correctness, speed and publication | Greedy gate passes on the pinned reference and quantized arm; TTFT, vision encode, prefill, decode and memory are recorded; user documents name exact weights |
| W7-CUDA | CUDA device path | A leased NVIDIA device runs the vision, merge and generation gates through the CUDA provider; the full-artifact arm uses a device/topology with enough memory |
| W7-ROCM | ROCm device path | `strix:gpu0` runs the HIP/ROCm provider gates through `rc run`; unsupported full-artifact residency is recorded as a memory blocker, never replaced by a CPU result |
| W7-VULKAN | Vulkan device path | `strix:gpu0` runs the Vulkan provider under RADV on the physical `AMD Radeon Graphics (RADV GFX1151)` device; the gate rejects llvmpipe or any CPU Vulkan device |

A fresh implementer owns each implementation wave from the committed spec. A
fresh reviewer inspects each immutable head, mutates every claimed guarantee and
restores the scratch tree byte-for-byte. The operator reruns the row gate before
integration. Correctable findings return to a fresh implementer; the operator
does not repair them.

## Tests to port

### RED-first focused tests

- `test_deepseek_v4_encoding`: all pinned encoder tests and exact error text or
  error category where C++ wording must differ.
- `test_deepseek_v4_image_processor`: grid/token counts, resize/pad, BF16 patch
  bytes, N-layout types/permutation, multiple images and placeholder mismatch.
- `test_deepseek_v4_vision`: ViT block and aligner stage numerics, 2-D RoPE axes,
  full bidirectional attention and BF16 memory format.
- `test_deepseek_v4_mm_forward`: sentinel replacement, image visibility,
  atomic-prefill refusal, no vision work on decode and text-only inertness.
- `test_deepseek_v4_mm_loader`: real config, complete pinned index, official
  storage formats, GGUF name map and named refusals. Includes: all 43
  `blk.N.exp_probs_b_vl.bias` are accounted for and loaded; the `deepseek4v`
  mmproj's 427 tensors map with none unaccounted; every other
  `clip.projector_type` still refuses by name.
- `test_deepseek_v4_mm_router_bias`: an image row takes `exp_probs_b_vl` and a
  text row in the same batch takes `exp_probs_b`; on a hash layer the image row
  takes `exp_probs_b_vl` while the text row takes `tid2eid` and no bias. Swapping
  the two biases must make this red, and it must not be observable only through
  generated tokens.
- `test_deepseek_v4_mm_window`: with `sliding_window = 128` and an image span
  longer than the window, every position inside the span is visible to every
  other position in it, and positions below the span start stay window-clipped.
  Restoring the plain window mask must make this red.
- `test_deepseek_v4_mm_e2e`: production `ModelRegistry::Forward` on the pinned
  image prompts, then real-checkpoint generated ids.
- `test_deepseek_v4_mm_server`: OpenAI multi-image request through the actual
  server surface, including PNG/JPEG data URIs and HTTP(S) media.
- `test_deepseek_v4_vision_device`: the same reduced-shape tower, aligner and
  merge cases run through CUDA, ROCm and Vulkan providers, with backend-specific
  tolerances derived from the CPU/oracle result and an assertion naming the
  physical device/provider.

Every test enters through the lowest production seam that can observe its
contract. W4 and later include a reachability mutation: remove the registered
production call site and confirm the focused gate fails.

## Gates

W0 is records-only and does not claim a permanent product gate. Its record
checkers must pass before the spec commit. Each implementation wave adds the
first behavioral command that can falsify its own scope.

Each implementation wave records its focused and full commands before it moves
to `ACTIVE`. W6 uses committed oracle and server harnesses so the exact
revision, images, prompts and sampling parameters are reviewable.

## Backend gate matrix

The operator schedules each device gate through resource-controller and records
the job id, selected device, backend build identity and contention state.
Long jobs set a maximum runtime. A missing toolkit is installed or staged inside
the leased worker as its usage sheet permits; it never authorizes a direct SSH
run.

| Path | Leased device | Required proof |
|---|---|---|
| CUDA | `dgx:gpu0` for the full model; `thor:gpu0` or `orin:gpu0` may run reduced device cases when their memory and architecture fit | CUDA provider selected, device buffers remain resident, reduced stage numerics pass, then the eligible real-artifact gate passes |
| ROCm | `strix:gpu0` | HIP build selects the ROCm provider, reduced stage numerics and memory-format checks pass on Radeon-8060S, and no CPU reference-tier fallback is reported |
| Vulkan | `strix:gpu0` | Vulkan build selects RADV GFX1151, not llvmpipe; reduced stage numerics and buffer residency pass through the Vulkan provider |

The Vulkan capability was measured under resource-controller job
`9eeefe15-1221-4dbf-938a-a0e1d18518bb`: Vulkan 1.3.275 exposed physical device
`AMD Radeon Graphics (RADV GFX1151)` with RADV/Mesa 25.2.8. ROCm and Vulkan may
use the same physical leased device in separate jobs and separate builds; the
providers are distinct gate results.

## Oracle evidence

`tools/oracle/deepseek_v4_vision_oracle.py` records the exact oracle revision,
package versions, model artifact identity, prompt, image hashes, sampling
parameters and output. Committed goldens include processor outputs, selected
vision/aligner stage tensors, first-step logits and generated ids. The first
oracle run repeats greedy generation enough times to determine whether strict
token-exact or a ratified distributional gate is valid. It does not choose a
weaker gate in advance.

## Performance axes

After correctness:

- image decode and preprocessing time;
- vision encoder and aligner time per image and for the two-image fixture;
- language prefill TTFT;
- steady cached decode tokens/s;
- peak and resident memory;
- concurrency 1 and the first supported concurrent batch.
- the same correctness, residency, TTFT and decode axes for each applicable
  CUDA, ROCm and Vulkan arm, labelled with the resource-controller device and
  job id.

The denominator is the pinned model-author runtime until vLLM implements the
model. When vLLM gains support, the row reconciles onto vLLM and reruns every
applicable axis in vLLM's production configuration. No apparent limit is called
a ceiling.

## Risks and decisions

1. **The oracle is large and unrun.** Static source agreement cannot promote it
   to gateable. Stop at the exact external resource if TP4 cannot be leased.
2. **The architecture string is shared with text.** Capability metadata is
   architecture-wide; tower construction and multimodal execution are
   config/input conditional. Text checkpoints must remain tower-free and
   byte-identical.
3. **Image attention is not only masked scatter.** Dropping the visibility
   extension yields plausible tokens and can evade token gates. Stage and index
   tests are mandatory.
4. **Chunked prefill can split an image.** The processor/scheduler marks image
   blocks atomic or refuses before forward. The model-level assertion remains a
   defense, not the first user-visible failure.
5. **The reference preprocessing test is missing.** Execute pinned code to
   generate evidence; do not infer expected pixels from PIL behavior.
6. **The official artifact cannot fit one GB10.** The pinned unsloth `UD-IQ1_S`
   pair is 77.65 GiB resident and does fit, so the single-device gate is now
   reachable. It is a completion dependency, not an optional optimization.
7. **Remote image fetching is security-sensitive.** Reuse the shared HTTP/TLS
   transport and its timeouts; do not shell out or add a DeepSeek-only fetcher.
8. **DSpark weights are present.** Account for them and keep the optional
   speculator's lifecycle separate. Silent activation or silent dropping is
   forbidden.
9. **The upstream repository is experimental.** Every source and artifact link
   uses the 40-hex pin. A force-push or replacement checkpoint triggers the stop
   condition below.
10. **Backend parity is explicit.** CUDA success cannot stand in for ROCm or
    Vulkan. Each provider receives its own leased build and execution result.
    A backend that cannot hold the complete artifact keeps that axis
    `PENDING` on measured memory while its reduced device path remains required.

## Stop conditions

- The pinned Hugging Face revision no longer resolves or its index/artifact
  identity changes: stop, record the new identity, and obtain a new pin decision.
- A complete oracle cannot build or run on an eligible leased topology: keep
  `gateable = no`, record the exact dependency or hardware blocker, and do not
  claim end-to-end support.
- The pinned unsloth revision no longer resolves, or a shard's SHA-256 changes
  under an unchanged name: stop and obtain a new pin decision. Re-quantization in
  place is why the revision and hashes are pinned rather than the repository id.
- The pinned quantized arm cannot preserve the released tensor set or fit an
  available gate device: the row remains incomplete; loader scaffolding that
  cannot generate from an image is not support.
- Resource-controller reports no matching healthy device or loses a worker:
  keep only that backend gate `PENDING`, record the controller/device state, and
  do not bypass the lease with direct SSH or substitute another backend.
- Current vLLM lands a complete implementation before W1: stop and rebase the
  design onto that exact vLLM revision rather than maintaining the model-author
  runtime as the mirror source.
- A wave requires bypassing `ModelRegistry::Forward`, the shared multimodal input
  types or the DeepSeek attention/KV seams: return `NEEDS_DECISION` with the
  unrepresentable behavior and the smallest seam extension.

## The language-side tensor delta is closed, and it is 43 names

Measured 2026-09-05 by range-reading all three `UD-IQ1_S` shard headers of
`unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF` at revision
`b977d3c0ea2da58dbc12ddae8fb8951a7b3854d0` and diffing the 1371 names against
`scripts/dsv4_gguf_manifest_names.txt`, the 1328-name manifest that
`scripts/check-dsv4-gguf-namemap.py` already pins for the TEXT artifact
`unsloth/DeepSeek-V4-Flash-GGUF`:

| Direction | Count | Names |
|---|---:|---|
| in vision, not in text | 43 | `blk.N.exp_probs_b_vl.bias`, `N` in 0..42 |
| in text, not in vision | 0 | — |

The two artifacts declare the same topology, so the counts are comparable:
`block_count = 43`, `expert_count = 256`, `hash_layer_count = 3`, and an
identical `attention.compress_ratios` array.

This is a completeness result, not a convenience. It says the vision checkpoint's
language half needs **nothing** from this port beyond the bias W3B loads, and
that the entire tower, aligner and sentinel group lives in `mmproj-BF16.gguf`,
which is W3A's scope. W4 therefore has no third unknown tensor family waiting for
it. `scripts/dsv4_vision_gguf_manifest_names.txt` is the committed fixture.

`check-dsv4-gguf-namemap.py` asserts EXACT set-equality against the text
manifest, so it cannot see this artifact at all today. Extending it to the vision
manifest is a semantic checker change and is owed below, with the measurement
above as its red-before input.

## Owed
- `ResidentWeight`'s device-staging arm in
  `include/vllm/model_executor/models/dense_attn_block.h` drops `q8_0_aligned`
  and `repacked` while guarding `elem_kn_repacked`, so the shared seam cannot
  carry a device-relevant storage layout for any model that inherits it. Found
  while repairing this row's own private copy of the same defect. It is a
  shared-seam gap outside this row, and
  [#2992](https://github.com/mudler/vllm.cpp/issues/2992) owns it.

- The first TP4 oracle run and committed evidence are owed by issue #2411 and W1.
- The unsloth GGUF arm's first load and generation, on the pinned revision and
  hashes above, is owed by issue #2411 and W3.
- The first `llama-cpp-dsv4vision` build and run is owed by issue #2411; the
  oracle file records `gateable = no` until then.
- `exp_probs_b_vl` is ACCOUNTED FOR in all three loader arms by W3B and LOADED
  in the two that materialize a tower, the GGUF arm and the EXL3 carried arm.
  The official dense safetensors arm accounts without materializing, exactly as
  it does for every other tensor, so its W2b residual covers this one too.
  Nothing selects the bias. Three behaviours stay owed by issue #2411 and W4,
  and row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` owns the wiring:
  per-token selection between the two biases in `deepseek_v4_moe.cpp`; the
  hash-layer replacement at forward time, where an image row takes
  `exp_probs_b_vl` while a text row takes `tid2eid` and no bias; and the
  non-causal image-span sliding-window change. The loaded bias is a staged slice
  until W4 lands, in the sense of `AGENTS.md` "Nothing lands dead".
- `scripts/check-dsv4-gguf-namemap.py` is owed the vision manifest. It generates
  1328 expected names and asserts exact set-equality against the TEXT artifact,
  so the 1371-name vision artifact fails it by construction and no gate covers
  the shipped vehicle's language half. `scripts/dsv4_vision_gguf_manifest_names.txt`
  is the committed fixture and the measurement above is the red-before input.
  The checker change is not made here because it is a semantic checker change
  and needs its own red-before evidence. Issue #2411 and W3 own it.
- CUDA, ROCm and Vulkan device-path evidence are owed by #2411 W7-CUDA,
  W7-ROCM and W7-VULKAN. Every run uses `rc`; a CPU fallback is not evidence for
  any of the three.
- The W2 vision tower and aligner are unreachable from a production entry
  point. `DeepSeekV4Vision`, its `Forward`, `VisionForward` and
  `AlignerForward` seams, `DeepSeekV4VisionRopeCosSin` and the
  `DeepSeekV4VisionCapture` type have no production call site: nothing in
  `include/vllm.h`, the loader, `ModelRegistry::Forward` or a registered server
  or command-line path constructs the class, and the stage goldens reach it by
  building it in the test. The only non-test file that includes the W2 header is
  `clip_mmproj_gguf.h`, for the `DeepSeekV4VisionConfig` and
  `DeepSeekV4VisionWeights` types W3A's reader fills, and that reader is
  unreached for its own reason below. Row
  `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` owns the wiring in W4, which
  routes the tower through the registered model forward, and issue #2411 tracks
  it. W2's own commit body claimed this entry was already here when it was not,
  which is the omission the W2 repair closes.
- W1 prompt encoding and image preprocessing remain unreachable from a
  production entry point. W4 wires them into the registered model forward, and
  W5 wires the runner, public ABI and OpenAI server for row
  `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`; issue #2411 tracks both
  waves.
- The W3A `deepseek4v` mmproj reader is unreachable for the same reason.
  `RefuseUnsupportedDeepSeekV4ClipMmproj`,
  `DeepSeekV4ClipMmprojVisionConfig`, `LoadDeepSeekV4VisionFromClipMmproj`,
  `DeepSeekV4ClipMmprojExpectedTensors` and
  `RefuseUnaccountedDeepSeekV4ClipMmproj` have no production call site: the one
  `clip` mmproj call site, `src/vllm/entrypoints/model_loader.cpp`, still calls
  the Qwen3-VL arm only, and that arm's refusal deliberately keeps rejecting
  `deepseek4v` so a DeepSeek projector cannot reach a Qwen3-VL reader. Row
  `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` owns the wiring in W4 and
  issue #2411 tracks it. The four sentinel vectors this reader returns
  (`image_start`, `image_end`, `image_pad`, `image_newline`) also have no
  consumer until W4 assembles the token block.
- The pinned `mmproj-BF16.gguf` has never been read by this code. W3A gates the
  name map, the metadata map and the four layout joins against a synthetic
  fixture built to the artifact's measured header; the real 934,462,656-byte
  file is owed by W3 together with the arm's first load and generation.
- DeepSeek-V4 DSpark remains owned by
  `MODEL-SPEC-deepseek-v4-dspark-deepseek-v4-for-causal-lm`; this row only
  accounts for and names its tensors.
- The FUSED `v.blk.{bid}.attn_qkv` mmproj arm is NOT IMPLEMENTED, and it is
  owed by issue #2411 and row
  `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`. `gguf-py/gguf/constants.py`
  at the pin spells V_ENC_ATTN_QKV `v.blk.{bid}.attn_qkv`, and nothing splits it
  for this family: `conversion/base.py` contains no occurrence of `qkv` at all,
  the only converter that splits a fused vision qkv is the model-specific
  `conversion/qwenvl.py`, and
  `conversion/deepseek.py::DeepseekV4FlashVisionModel.modify_tensors` splits
  `mlp.w1` only. So a projector converted by the pinned oracle's OWN
  `convert_hf_to_gguf.py` carries `v.blk.N.attn_qkv.{weight,bias}` and is 299
  tensors at depth 32, and this build refuses it BY NAME.
  `RefuseUnsupportedDeepSeekV4ClipMmproj` states that the fused arm is not
  implemented and points at #2411, so no user reads the unaccounted-tensor
  refusal and re-converts a file that is already correct. THIS DOES NOT BLOCK
  THE SHIPPED VEHICLE: `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF` carries the
  SPLIT form, 427 tensors, verified against its own header. It DOES block
  converting the checkpoint with the oracle's own script, which is a
  quant-matched denominator W6 may need.
- The vision `rope_theta` is unkeyed on BOTH sides and is owed by issue #2411.
  `tools/mtmd/clip.cpp` hardcodes `10000.0f` for this projector and
  `conversion/deepseek.py` defaults `vision_rope_theta` to 10000.0 without
  writing a key. That is correct for this artifact, and the same converter
  asserts `vision_max_n_token == 384` and `vision_max_wh_ratio == 8` while
  asserting NOTHING about the theta, so a future variant with a different one
  would be read silently wrong — by llama.cpp as well as by this reader. No
  code change is made here, because there is no key to read.
- Four `clip.*` keys the real `mmproj-BF16.gguf` carries are read by nothing in
  this tree yet, and they are the PREPROCESSOR CONTRACT that W4 and W5 owe
  under issue #2411: `clip.vision.image_size = 672`,
  `clip.vision.image_mean = [0.5, 0.5, 0.5]`,
  `clip.vision.image_std = [0.5, 0.5, 0.5]` and
  `clip.vision.image_min_pixels = 147456`. W1's preprocessor currently takes
  these from its own configuration rather than from the projector that shipped
  with the weights.
- The reader's `general.alignment` fallback is never exercised. The fixture's
  builder always writes the key, and the real artifact carries no alignment key
  at all, so the default-32 path the shipped file actually takes is the one path
  the gate does not cover. Widening the fixture is owed by issue #2411 and W3;
  it needs a change to the shared `tests/vllm/gguf_builder.h`, which every GGUF
  test uses, so it is not made inside a W3A repair.

### W3A evidence

W3A adds the `deepseek4v` arm to `src/vllm/model_executor/models/clip_mmproj_gguf.cpp`,
the reader that already carries `qwen3vl_merger`. The container is anchored at
the secondary oracle `llama-cpp-dsv4vision`, release `b10766` =
`9400c8946e4da5e7694f2c26d6d4e50e14b690fa`. The anchors were read from the diff
that introduces `tools/mtmd/models/deepseek4v.cpp` as blob `ffe8f59d9997`, which
is the blob that path holds at that pin, so the citations are the pin's own
bytes rather than a moving pull-request head.

The artifact's header was re-read on 2026-09-05 over an HTTP range request for
its first mebibyte, without downloading the 934,462,656-byte file. It reports
GGUF v3, 427 tensors and 27 keys: `clip.projector_type = deepseek4v`,
`projection_dim = 4096`, `patch_size = 14`, `embedding_length = 1024`,
`feed_forward_length = 2816`, `block_count = 32`, `attention.head_count = 16`,
`attention.layer_norm_epsilon = 9.999999974752427e-07`, `use_silu = true`,
`projector.scale_factor = 3` and `image_min_pixels = 147456`. Its 2-D linear
weights are BF16; every bias, every norm weight, `v.patch_embd.weight` and the
four sentinel vectors are F32. The reader's own enumeration returns 427 names at
`depth = 32`, which the focused gate asserts.

Four layout mismatches separate what the file stores from what W2 consumes, and
each is a silent wrong answer rather than a crash. `attn_q` / `attn_k` /
`attn_v` are stored separately IN THIS FILE and fuse in that row order, which is
the order `deepseek_v4_vision.cpp` slices back out with `RowSlice`. The split is
a property of the shipped artifact and not of the family: the pinned
`convert_hf_to_gguf.py` emits the FUSED `v.blk.{bid}.attn_qkv` instead, which
`## Owed` records as an unimplemented arm. `ffn_gate` and
`ffn_up` are stored separately and concatenate gate-first, which is the half
`vt::SiluAndMul` applies SiLU to and the half the pinned converter's
`gate, up = data_torch.chunk(2, dim=0)` took. `v.patch_embd.weight` is a conv2d
view of an `nn.Linear` over an `F.unfold`, so its flattening back to
`[hidden, 3*patch^2]` is the identity in `[channel, dy, dx]` order rather than a
permutation. The file's f32 storage of every bias and of the patch embedding is
llama.cpp's small-tensor convention, so those narrow to the model dtype while
the RMSNorm weights stay f32, exactly as W2's contract states.

The gate started RED. `cmake --build build-w3a --target test_deepseek_v4_mmproj -j 3`
failed with 99 compiler errors, every one naming a symbol the reader did not yet
have: `RefuseUnsupportedDeepSeekV4ClipMmproj is not a member of vllm; did you
mean RefuseUnsupportedClipMmproj?`, and the same for
`LoadDeepSeekV4VisionFromClipMmproj`, `DeepSeekV4ClipMmprojExpectedTensors`,
`RefuseUnaccountedDeepSeekV4ClipMmproj`, `DeepSeekV4ClipMmproj` and
`multimodal::DeepSeekV4VisionConfig`.

After the change the focused gate passes 13 cases and 999 assertions, and
`ctest --test-dir build-w3a -R deepseek_v4_mmproj --output-on-failure` reports
1/1 on a Release CPU build with `-DVLLM_CPP_CUDA=OFF`.

Five production-source mutations prove the gate detects each claimed guarantee.
Permuting the fused order to q, v, k reddens case (a) with 288 failed
assertions. Swapping `ffn_gate` and `ffn_up` reddens case (b) with 192.
Reordering the patch flattening to `[dy, dx, channel]` reddens case (c) with 80.
Routing the RMSNorm weights through the model-dtype narrowing reddens case (d)
with 45. Replacing the `clip.vision.attention.layer_norm_epsilon` read with the
W2 default of 1e-6 reddens the config case with 1, which is why the fixture's
epsilon is 1.5e-5. Each mutation restored
`src/vllm/model_executor/models/clip_mmproj_gguf.cpp` byte-for-byte, verified by
`sha256sum -c` against
`465c762530030ff31b018080aa0020c7bbfd950d7b86f7489a3adee42bc1d0e5`.

The Qwen3-VL arm is deliberately unchanged. `RefuseUnsupportedClipMmproj` still
refuses `deepseek4v`, because `src/vllm/entrypoints/model_loader.cpp` goes
straight from that refusal into `LoadQwen3VLVisionFromClipMmproj`, and widening
it would route a DeepSeek projector into the Qwen3-VL reader. A case in the new
gate asserts that refusal still fires and still names both projector types.
`test_clip_mmproj_gguf`, `test_gguf_mmproj_reach`, `test_gguf_accounting_reach`
and `test_qwen38_27b_gguf_manifest` pass 4/4 on the same build.

The reader is not reached from production. `## Owed` names what is unreached,
the row that owns the wiring and issue #2411.

## Now

`ACTIVE`. W1 and W2 have landed on the row branch, and this spec was amended on
2026-09-05: the quantized vehicle is now the pinned
`unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF`, `llama-cpp-dsv4vision` is
registered as a runnable reference for it, and two language-side vision
behaviours the original spec missed (`exp_probs_b_vl` and the non-causal
image-span window) are specified and owed. W2 adds the standalone
32-layer-capable ViT and the downsample-3 aligner as a config-driven composition
over public `vt` operations. Neither wave is reachable from production: W3 owns
weights, W4 owns the registered forward and image-span visibility, and W5 owns
the runner, public ABI and server.

### W1 evidence

The repair tests ran before each repair. The Pillow matrix returned BF16 words
`15885` and `48942` instead of `15877` and `48944` for the pinned 3x5 seed-0
case. Falsey image-source forms, BF16-only output, configuration validation,
checked grid arithmetic and the sized RGB API each failed their selected test
or build for the intended reason. Removing final N-layout alignment padding
made its new 2x2 case fail.

The second repair added token-budget cases for `6` and `8`. Both cases accepted
the invalid configuration before the repair. An allocation probe measured two
allocations and 36 bytes for the 12-byte identity image. The repaired path makes
one 24-byte allocation for its BF16 output. A `max_image_tokens` value of `9`
processes the minimum 1x1 image.

The resize implementation ports Pillow 12.1.1
`src/libImaging/Resample.c::{precompute_coeffs,normalize_coeffs_8bpc,
ImagingResampleHorizontal_8bpc,ImagingResampleVertical_8bpc}` and
`PIL/ImageOps.py::{contain,pad}`. Seven deterministic oracle fixtures cover
ordinary upsampling, downsampling, aspect padding and the direct wide-image
branch. They compare all BF16 words through fixed hashes and pin selected words
explicitly. The exact wide threshold fixture has a 3x6 image and a ratio of
`2`. Its complete BF16 output hashes to `0xab9bbef0bbb70c6a`.

The task repair started red: number, boolean, array and object tasks rendered as
ordinary prompts, and an invalid string before another user transition returned
without validation. A null task incorrectly suppressed retained assistant
reasoning. The image-boundary regression accepted one BF16 feature for
`patch_size = 2`, whose required feature width is `12`.

A fresh Release CPU build used
`cmake -S . -B build-repair-clean -G Ninja -DVLLM_CPP_BUILD_TESTS=ON
-DCMAKE_BUILD_TYPE=Release`, then built both W1 targets. The command
`ctest --test-dir build-repair-clean -R
'^test_deepseek_v4_(encoding|image_processor)$' --output-on-failure` passed
2/2 tests.

Sixty-two independent production-source mutations cover every W1 encoding and
processor guarantee. The ten repair mutations include the earlier token-budget
floor, exact wide comparison, identity-buffer and historical-thinking
mutations. The six latest mutations inverted the task type and membership
guards, replaced all six task tokens, disabled the action assistant transition,
treated a null task as present and inverted the patch-feature-width guard. Each
selected focused test went red. Each mutation restored the source
byte-for-byte.

The default multi-turn thinking case emitted `<think>` before historical
assistant content instead of the pinned `</think>`. The repaired transition
keeps `<think>` only when thinking is retained or the message is at or after the
last user. The final focused gate passed with source SHA-256
`9f043f826e38803aca19da29e92aa5103cedb70f8792fe4693bf81d695804886`,
header SHA-256
`6c4224c11280430a41aeb4c50b31c3ded921af44f8a0d328fd2164216bebb6f1`,
encoding-test SHA-256
`35bb6ba50cfc7eda8c87ca6b0aaed04c826ef8b96231c8601b96d55cd252da7d`
and processor-test SHA-256
`0d0d44a2a741e5b52d1acc41c9530f0addb8694e27f03448fb879d25c25c7ade`.

### W2 evidence

`heads2_depth2` and `heads4_depth1` in
`tests/parity/goldens/deepseek_v4_vision/goldens.json` were generated by
`scripts/gen-deepseek-v4-vision-goldens.py`, a direct transcription of
`inference/vision.py` at pin `86f746b36186f0e567729a5c06a8c918caba82a9` under
torch `2.11.0+cu130`. They cover the 2-D RoPE cos/sin tables, patch embedding,
per-block outputs, the final norm, the unfold order, the aligner hidden and
GELU stages, plus head-geometry, dtype and contract refusals.

`cmake --build build-w2 --target test_deepseek_v4_vision -j 4` then
`ctest --test-dir build-w2 -R deepseek_v4_vision --output-on-failure` passed
1/1 on a Release CPU build.

`check-attention-rung-consistency` was red on this tree: the tower's
`vt::Attention` call carried its `VT-ATTN-NAIVE:` reason 46 lines above the
call, and the checker reads the call line or the 20 lines above it. The reason
now sits on the call. The checker reports 9 of 9 marked sites, and its mutation
suite `tests/scripts/test_check_attention_rung_consistency.py` passes 39/39.

### W2 repair evidence

A fresh reviewer mutated W2's claimed guarantees and returned seven findings.
All seven are repaired. Each mutation below was applied to the tree, rebuilt
(ninja always did work, never zero steps, so no result is a stale build), run,
and restored byte-for-byte; the source file's sha256 after every restore is
`6006b685da095ade85c2c353083860f81b04f183f3d0491bacc85899617b3aa0`, and the tree
was rebuilt and re-run green after each one.

**Two reduced fixtures had degenerated the axes they were believed to gate.**
`heads2_depth2` and `heads4_depth1` are both head_dim 4, so `rope_dim` is 2 and
`get_vision_cos_sin` has EXACTLY ONE frequency per axis at exponent
`2*0/rope_dim = 0`. `inv_freq[0]` is therefore `theta**0 = 1.0` for every theta,
and both pinning `rope_theta` to a literal 10000.0 and halving the exponent
denominator left every golden byte unchanged. Separately, every fixture grid
((2,5), (3,3), (3,4)) aligns to ONE merged row at ratio 3, where a row-major and
a column-major walk of the merged grid are the same sequence, so swapping the
`block_row` and `block_column` loops was invisible; the dedicated unfold case
could not catch it either, because it builds its expectation with the same loop
nesting as the implementation.

`heads1_headdim16_theta7919` closes both. head_dim 16 gives four frequencies at
exponents 0, 1/4, 1/2 and 3/4, its theta is neither the default nor either other
fixture's, and its grids 4x5 and 7x4 merge to 2x2 and 3x2.

**The fixtures were regenerated with the committed generator, which is the only
available option, and that limit is stated rather than hidden.** There is no
local checkout of `86f746b36186f0e567729a5c06a8c918caba82a9` and no network
access to it, so the formulas cannot be re-derived from source, and a
transcription error shared between the generator and a new fixture would NOT be
caught by adding fixtures from that generator. What could be checked was: the
local torch is `2.11.0+cu130`, exactly the version the fixture records, and
re-running the generator before the change reproduced the committed goldens
byte-for-byte. The regenerated file is a pure insertion of 7448 lines, so the two
original fixtures are untouched.

**The row order is confirmed by a second, independent oracle, so the code was
correct and merely ungated.** llama.cpp release `b10766` =
`9400c8946e4da5e7694f2c26d6d4e50e14b690fa` (oracle `llama-cpp-dsv4vision`) maps
merged cell (row r, column c) to aligner output row `r * n_llm_w + c` in
`clip.cpp`'s `set_input` for `PROJECTOR_TYPE_DEEPSEEK4V`, and its graph in
`tools/mtmd/models/deepseek4v.cpp` (blob `ffe8f59d9997` at that pin) reaches the
same order through `ggml_im2col` over a `[x, y, n_embd]` tensor reshaped
`[ne0, ne1*ne2]`, which flattens `[OW, OH]` with OW fastest. The new row-order
case takes its destination index from that formula rather than from our loop
nesting, so it is not a second copy of the implementation.

**One tolerance was changed. It is a correction AND a concession, and the first
telling of it said only the first half.** SUPERSEDES the W2 repair evidence
committed at `d825a5133`, whose claim of "a correction rather than a concession"
is withdrawn here; the numbers below replace its constant and its per-case
comparison.

The `gelu` stage carried a declared bound of 0.01 that was LOWER than the 0.016
allowed for the `aligner_hidden` buffer feeding it. That ordering is not
derivable, and that part of the original observation stands. What was wrong was
the constant and the accounting.

THE CONSTANT. GELU(x) = x*Phi(x), so GELU'(x) = Phi(x) + x*phi(x) and
GELU''(x) = phi(x) * (2 - x^2), which is zero at x = sqrt(2). Therefore
`sup|GELU'| = Phi(sqrt2) + sqrt2*phi(sqrt2) = 1.1289041452` at x = 1.41421, and a
brute-force sweep of [-10, 10] at 1e-5 agrees to seven figures. The committed
value of 1.084 "attained near x = 1.5216" was wrong twice: 1.0833155 is
GELU'(1.0), the derivative at 1 rather than at the stationary point, and
GELU'(1.5216) is 1.1266919, so the stated value and the stated maximizer did not
agree with each other either. GELU can amplify the error it is handed by 12.9%,
not by 8.4%.

THE ACCOUNTING. Measured per case, `aligner_hidden` to `gelu`: 0.0078125 to
0.0078125, 0.0078125 to 0.0078125, 0.015625 to 0.00878906, 0.0078125 to
0.00390625, and 0.0136719 to 0.0117188. Every case attenuates and none reaches
the ceiling. Against the 0.01 it replaced, `max(0.004, 1.1289042 * the case's own
aligner_hidden error)` is TIGHTER for the three cases at 0.0078125, which give
0.008820, and LOOSER for the two above them: 0.015625 gives 0.017639, a 76%
widening, and 0.0136719 gives 0.015434, a 54% widening. The first telling
reported the three that tightened and not the two that widened.

WHAT THE WIDENING COST, MEASURED. A `vt::GeluErf` that scales its output by
1.004f -- one bf16 ulp at these magnitudes -- when and only when it is called on
more than one row is a real defect. It is invisible to the single-row
`gelu_probe`, because that probe never enters the branch, and under the derived
bound the whole suite stayed green at 15 of 15 cases and 7404 of 7404 assertions.
Under the 0.01 it replaced, that same mutation reds two assertions rather than
one: `heads4_depth1 / 3x4` at 0.0117188, a case that caught it BEFORE the change,
and the new `heads1_headdim16_theta7919 / 7x4`. Reverting the bound to 0.01
without the mutation reds exactly one assertion, the new fixture's, at 0.0117188
-- so a new fixture failing the old bound is what drove the change.

WHY THE WIDENING IS KEPT, AND WHAT PAYS FOR IT. A stage bound below its own
input's bound is not derivable, and `heads1_headdim16_theta7919 / 7x4` is handed
0.0136719 by `aligner_hidden`, so no absolute ceiling at or below 0.01 can stand
here. The coverage the widening removed is restored at the observable that owns
it: the exact-erf probe now runs at `aligned_rows(downsample_ratio + 1, 1) = 2`
rows as well as at 1, comparing bit-exactly against the pinned golden on every
row. Red before: the 1.004f multi-row mutation, which was green on the whole
suite and now reds `DeepSeek-V4 aligner uses exact erf GELU` at `rows := 2`.
Green after: 15 of 15 cases and 7407 of 7407 assertions with the tree restored.
`aligner_hidden` keeps its absolute cap, so the stage stays transitively bounded
at 0.016 * 1.1289042 = 0.0180625.

Every stage upstream of GELU on the case that first failed is at or below what
the pre-existing fixtures already produce: patch 0.00195312 against 0.004, vision
and unfold 0.015625 against 0.024 where an existing case reaches 0.0234375, and
aligner_hidden 0.0136719 against 0.016 where an existing case reaches 0.015625.
The new geometry is not worse anywhere.

**A dtype that is too wide, and per-layer scratch, both needed observables that
no value gate provides.** Widening the attention-output buffer to f32 was fully
green, exactly as `AGENTS.md` warns under "Inherit vLLM defaults". The forward
now reports the dtype of every internal scratch buffer it allocates, in
allocation order, through a capture field production never sets, and the test
asserts the exact sequence and the count of f32 entries. The two f32 entries are
the rotary pair and keep their reason.

**That recorded list is a hand-maintained mirror, and on its own it does NOT
hold the guarantee `34f175fb4` claimed for it.** SUPERSEDES that commit's "a new
wide buffer cannot be added without the case failing", which is withdrawn:
`RecordScratch` is called by hand at each allocation site, so a buffer that does
not call it is invisible to the list. A fresh review hoisted an f32 attention
buffer and round-tripped the attention output through `CastF32`/`CastBf16` --
identical values, twice the bytes on the model path -- with no `RecordScratch`
call, and the whole suite stayed green at 15 of 15 cases and 7407 of 7407
assertions. `f32_entries == 2` counts recorded entries only; the pool-slope case
measures traffic per layer, which a hoisted buffer does not change; and
`at_deep.misses == at_shallow.misses` is an equality across depths that a
constant +1 satisfies. Reproduced here rather than taken from the report.

The list is now bounded by something the code cannot drift from: the bytes the
pool hands one Forward. Every `DBuf` in the forward draws from
`vllm::Pool(backend)` whether or not anything records it, so one Forward from a
drained pool prices the whole model path in two numbers. Measured on this tree at
fixture 0, deterministic over three runs and at both depths: 13 driver
allocations totalling 2680 class-rounded bytes. Under the review's mutation, 14
and 3000, and 3000 - 2680 = 320 is exactly the [10, 8] f32 buffer it added. The
gate is a CAP rather than an equality, because a pool block is class-rounded and
another backend may serve the same forward from fewer blocks, while every way of
widening the model path can only push it up. Red before: the mutation is green on
the whole suite and now reds both assertions at `14 <= 13` and `3000 <= 2680`.
Green after: 15 of 15 cases and 7409 of 7409 assertions with the tree restored
and verified by SHA-256.

What the cap does NOT see is a new buffer the pool serves from a block that was
already free, which adds no driver allocation and no retained bytes. That is
narrower than the withdrawn claim and is stated rather than assumed.

For the per-layer scratch the review proposed bounding pool `misses` after a
single Forward independently of depth. That bound is true but CANNOT see the
defect, and this is measured rather than argued. One Forward from a drained
pool, hoisted against un-hoisted:

| depth | hoisted | un-hoisted |
|---|---|---|
| 2 | 17 gets (13 misses, 4 hits) | 18 gets (13 misses, 5 hits) |
| 4 | 21 gets (13 misses, 8 hits) | 24 gets (13 misses, 11 hits) |
| 8 | 29 gets (13 misses, 16 hits) | 36 gets (13 misses, 23 hits) |

`misses` is 13 in BOTH forms at every depth: the fixed working set is identical
and the pool serves every extra request from its own free list. The observable
that separates them is pool GET traffic PER LAYER, 2 hoisted against 3
un-hoisted. The gate measures one Forward at two depths and asserts that slope is
2, the two buffers `UnquantizedMlpGateUpMethod::Apply` legitimately owns. The
depth-independent `misses` bound is kept beside it because it is true, not
because it can see this.

**`BorrowResidentWeight` had re-introduced the #2031 marker loss.** It copied
dtype, rank, shape, `nk`, bytes and `d_dev` and dropped `repacked`,
`q8_0_aligned` and `elem_kn_repacked`. That is the defect `main` fixed at
`7a937db8a` in the shared `dense_attn::ResidentWeight`, where an i8mm-interleaved
`block_q8_0x4` buffer (136-byte blocks) reached the quant GEMM flagged as flat
`q8_0` (34-byte blocks), decoded to NaN, then all-zero logits, then token id 0,
with nothing logged because the `lm_head` GEMM swallowed the NaN. The markers are
now PROPAGATED rather than refused: a fail-closed check would remove the CPU
i8mm fast path instead of fixing the loss. It is host-conditional
(`vt::cpu::QuantRepackActive()` is true only on aarch64 i8mm) so no golden can
move here, and W3A's mmproj reader is what makes it live rather than latent.

**`ValidateTensor` has 21 call sites, not 15, and both groups are now driven.**
Fifteen are the weight checks; five more sit behind `ValidateCaptureTensor`,
whose entire body could be replaced by a no-op with the suite staying green,
because the stage goldens pass CORRECT captures and exercise only the happy
path. Neutering that helper now reds 13 of the 14 capture rows. The fourteenth,
the block-capture count, stays green under that mutation and correctly so: it is
a separate check in `ValidateVisionIo` rather than a `ValidateCaptureTensor`
call. Those refusals are gate-facing rather than production-facing, since
production passes nullptr and copies nothing, which is exactly why they needed
driving: a capture contract nothing checks lets a future parity gate read a
wrongly shaped buffer and compare whatever is in it.

**Both coverage guards were proved non-vacuous rather than assumed to be.**
A guard that passes because it asserts nothing is the same failure as the
degenerate fixture it exists to prevent. Setting the new fixture's theta back to
the 10000.0 default and regenerating reds the frequency guard at `0 >= 1`, and
replacing its grids with a single (2,5) reds the row-order guard. The tree was
restored and re-run at 14 of 14 cases and 7376 of 7376 assertions after both.

**W4 must size against the geometry cache.** The `IndexSelect` gather index is
`aligned_rows * hidden_size * downsample_ratio^2` i32 values per cached geometry.
At the production `hidden_size` 1024 and ratio 3, a 73x73 patch grid gives
`aligned_rows` 625 and an index of 5,760,000 i32 = 23.04 MB, and
`kGeometryCacheCapacity` is 8, so a full cache holds 184.32 MB of gather indices
alone. The other two per-geometry tensors are small beside it: the f32 RoPE cache
is 1.36 MB and the positions vector 21.3 kB at that grid. W4 owns whether eight
distinct geometries is the right capacity for the image sizes the server admits,
and whether the index should be computed rather than cached at that size.

**One gap stays open and is not this row's to close.** `ResidentWeight`'s
device-staging arm returns `MakeTensor(w.d_dev.get(), ...)`, which carries no
markers at all, so `q8_0_aligned` cannot reach a CUDA Q8_0 GEMM through that arm
for ANY model that uses the shared helper. That is a shared-seam gap in
`dense_attn_block.h`, outside this repair's scope, and it is recorded here rather
than repaired.

### W3B evidence

The artifact was read again before the wave started, not taken from the brief.
An HTTP range request over the first 14 MB of
`unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF`
`UD-IQ1_S/DeepSeek-V4-Flash-Vision-Exp-UD-IQ1_S-00001-of-00003.gguf` printed 72
key-value pairs and 43 tensors on 2026-09-05. The tensors are exactly
`blk.0..42.exp_probs_b_vl.bias`, each F32 `[256]`, and nothing else. The same
header carries `deepseek4.hash_layer_count = 3`, `deepseek4.block_count = 43`,
`deepseek4.expert_count = 256`, `split.tensors.count = 1371` and
`split.count = 3`.

llama.cpp PR #28154, at oracle `llama-cpp-dsv4vision`, is the reference for the
shape of the change. Its converter maps `ffn.gate.bias_vl` to
`blk.{bid}.exp_probs_b_vl`, drops `ffn.gate.bias` on every layer below
`num_hash_layers`, and creates `ffn_exp_probs_b_vl` OUTSIDE the hash branch with
`TENSOR_NOT_REQUIRED`. W3B mirrors the optionality and the every-layer scope. It
does NOT mirror the selection, which that PR makes per ubatch on
`ubatch.embd != nullptr`; the spec's `## Port map` requires a per-token rule and
W4 owns it.

RED first. `tests/vllm/models/test_deepseek_v4_mm_loader.cpp` failed 5 of its 6
cases before the loader changed. The GGUF cases threw
`deepseek-v4 gguf loader: LEFTOVER tensor not covered by the blk.N.* name map:
blk.0.exp_probs_b_vl.bias`. The EXL3 safetensors cases threw
`deepseek-v4 exl3 loader: checkpoint tensor no arm routes:
layers.0.ffn.gate.bias_vl`, so a vision checkpoint was REFUSED by that arm
rather than merely unaccounted. The official dense arm read
`CHECK( 112 == 114 )`: it accepted the two extra tensors and counted neither.
The sixth case, `dsv4 TEXT GGUF: the absent vision bias is accepted and changes
nothing`, passed before the change and after it.

Green after. `ctest --test-dir build-w3b -R test_deepseek_v4_mm_loader
--output-on-failure` passes on a Release CPU build configured with
`-DVLLM_CPP_CUDA=OFF`, and the test binary reported 6 of 6 cases and 83 of 83
assertions AS THIS WAVE LANDED. The W3B review repairs later took the same suite
to 9 cases and 105 assertions; `### W3B repair evidence` below carries the
current figure, and this paragraph is kept as the record of what W3B itself
measured rather than silently restated.

The inertness claim was mutated rather than read. Removing the optionality from
the GGUF arm, so `exp_probs_b_vl` is taken unconditionally, made
`dsv4 TEXT GGUF: the absent vision bias is accepted and changes nothing` the one
red case, with `gguf: no tensor named "blk.0.exp_probs_b_vl.bias"`. The source
was restored byte for byte afterwards; its SHA-256 is
`794c00f7557bbe71c858e82f0e85016475a7937264f5a93755e35705e2f070c2` before the
mutation and after the restore.

The text checkpoint's inertness was also checked outside this suite. The eleven
DeepSeek-V4 targets a Release CPU build can run --- `scaffold`, `moe`, `forward`,
`gguf_load`, `mtp_inventory`, `exl3_loader`, `mm_loader`,
`exl3_device_residency`, `exl3_forward`, `exl3_forward_loop_arm` and
`paged_equiv` --- all pass. `test_deepseek_v4_gguf_load` is the one that reads a
text `deepseek4` file end to end.

The two biases are filled from different functions in every fixture, so a loader
that routed one into the other's slot would still be caught. A gated layer's
`gate_bias` and `gate_bias_vl` are asserted to differ, and the hash-layer case
asserts that layer 0 carries `tid2eid` and `e_score_bias_vl` and an EMPTY
`e_score_bias`.

Nothing selects the loaded bias. The commit body names it unreached, names the
owning row and issue #2411, and `## Owed` above lists the three behaviours W4
owns.

### W3B repair evidence

A fresh review of W3B (`ebca4db83`) returned four findings. This section records
what each repair changed, the red result that was captured before it, and the
green result after it. The counts here supersede the `6 of 6 cases and 83 of 83
assertions` figure recorded above for `test_deepseek_v4_mm_loader`.

**F1 (blocking): the EXL3 carried arm asserted a slot, not the bytes.** The case
`dsv4 vision safetensors: the EXL3 carried arm routes and loads gate.bias_vl`
checked the width of `gate_bias_vl`, its non-emptiness, and `text[i] != vl[i]`.
Two defects pass all three. A slot filled with zeros keeps its width, and a swap
of the two biases keeps them unequal. The case now asserts every element of
`gate_bias_vl` on every layer, and every element of `gate_bias` on the gated
layer, against `dsv4_exl3_fixture::CarriedValue` for the tensor's own name.

The repair is a test repair. The loader was correct, and no product line changed
for this finding.

Both reviewer mutations were reapplied to `deepseek_v4_weights.cpp` and both are
now red. Replacing the read with `carried.Account(name)` plus an all-zero
`assign` gives four failures of the form
`CHECK( 0 == Approx( -0.999559 ) )` at layers 0 and 1. Transposing the two
biases on the gated layers gives four more,
`CHECK( 0.0988742 == Approx( -1.24119 ) )` on `gate_bias_vl` and
`CHECK( -1.24119 == Approx( 0.0988742 ) )` on `gate_bias`. The source was
restored byte for byte after each one; its SHA-256 read
`794c00f7557bbe71c858e82f0e85016475a7937264f5a93755e35705e2f070c2` before the
first mutation and after each restore, which is the same value the W3B evidence
above records.

**F2 (medium): the GGUF arm validated no width.** `V4GgufCtx::Vec` validates the
residency the policy elected and the role a tensor was routed under. It
validates no geometry, so a `[E-1]` router bias published under an unchanged name
loaded in silence and would be indexed by expert id, which reads past the end of
a short host `std::vector<float>`. The new `V4GgufCtx::Vec1D` takes the expected
width and refuses. This mirrors what the safetensors arm already gets from
`carried.Float(..., {ne})` and what `glm5_next_loader.cpp` and
`glm_moe_dsa_loader.cpp` already get from `LoadVecF32(g, name, e)`.

`Vec1D` reads that width from the FILE HEADER and refuses before the value is
materialized. Mutation: deleting the `VT_CHECK` makes both `NARROW` cases fail
together against an empty message, four assertions, which is the red the guard
was introduced against.

**The ORDERING is not gated, and the reason `e21dd054e` gave for it is wrong.**
SUPERSEDES that commit's account. A fresh review replaced `Vec1D` with the
materialize-first form -- `OwnedTensor t = Vec(name, role)` and then a check on
`t.rank` and `t.shape[0]`, with an identical message -- and
`test_deepseek_v4_mm_loader` stayed green at 9 cases and 105 assertions.
Reproduced here rather than taken from the report.

The finding is real and its remedy is not a new gate, because the danger the
commit named does not exist. `e21dd054e` said an absurd declared width would
surface as a failed allocation and take the machine down. It cannot:
`GgufFile::Open` in `src/vllm/model_executor/model_loader/gguf_reader.cpp`
refuses a tensor whose byte size overflows and then refuses any tensor span that
leaves the data section, so a header declaring four billion elements is refused
by name at Open and never reaches `Vec1D` at all. The real cost of
materialize-first is dequantizing a tensor whose file bytes already fit, which is
at most about 4x the bytes on disk for a Q8_0 vector and 1x for the f32 these two
biases are.

So the ordering is a preference for refusing early, not a correctness bound, and
no observable separates the two forms. Writing a case that pretends otherwise
would be the failure this repair exists to correct. The comment in
`deepseek_v4_weights.cpp` now says this, and nothing is owed.

**Both router biases are now checked, not only the vision one.** The text
`exp_probs_b.bias` beside it carried the identical weakness. It is a
long-standing gap rather than a W3B regression, and repairing one while leaving
its neighbour would leave the two to drift the first time either is touched. The
widened scope is stated here and in the commit body rather than left silent.

Red before: two new cases build a file whose KV declares `expert_count` and
whose tensor is one narrower. `LoadDeepseekV4FromGguf` raised nothing, so both
`CHECK(msg.find(...))` assertions failed against an empty message. Green after,
the refusals read

```text
vt: deepseek-v4 gguf: blk.0.exp_probs_b_vl.bias must be a 1-D [4] vector
(n_routed_experts), got rank 1 first dim 3
vt: deepseek-v4 gguf: blk.1.exp_probs_b.bias must be a 1-D [4] vector
(n_routed_experts), got rank 1 first dim 3
```

**F3 (low): per-layer optionality is deliberate, and is now executable.** No arm
requires the bias to be present on all layers or on none. The decision is to
MIRROR the oracle rather than to enforce the dichotomy. llama.cpp declares
`ffn_exp_probs_b_vl` with `TENSOR_NOT_REQUIRED` for each layer independently
(`src/models/deepseek4.cpp`, PR #28154 at the `llama-cpp-dsv4vision` pin), so a
partially converted file loads there. Refusing a file the oracle accepts is a
divergence that needs its own justification, and this one has none: the empty
slot is a state the consumer must already handle, because a text checkpoint
presents it on every layer.

The prose in W3B that reads "on every layer of a vision artifact and on no layer
of a text one" describes the two PUBLISHED artifacts. It is not a constraint the
loader enforces, and the new case
`dsv4 PARTIAL GGUF: the vision bias is optional PER LAYER, as in llama.cpp`
pins the behaviour. A later change to an all-or-nothing refusal is then a red
test somebody has to argue with, instead of a silent change of contract.

**F4 (record): the artifact the shard belongs to.** Section 1 above opened by
calling it "the unsloth text GGUF's first shard". The shard belongs to the
language half of the VISION repository. Section 1 now names the repository and
states that a text checkpoint carries none of these tensors, which is the fact
the optionality design rests on.

Gate after the repair, on a Release CPU build configured with
`-DVLLM_CPP_CUDA=OFF`. `ctest -R 'deepseek_v4_(mm_loader|gguf_load|exl3_loader|
moe|forward)'` passes 5 of 5. `test_deepseek_v4_mm_loader` reports 9 cases and
105 assertions, up from 6 and 83. The other four are unchanged at their reviewed
values: `gguf_load` 19 cases and 1056 assertions, `exl3_loader` 22 and 613,
`moe` 12 and 716, `forward` 6 and 34.

Nothing in this repair selects the loaded bias. W4 still owns the per-token
choice, the hash-layer replacement and the image-span window, and `## Owed`
above still lists them.

### W3A repair evidence

A fresh review of W3A returned six findings. The four layout derivations it
checked -- the q,k,v fuse order, gate-then-up, the identity patch permutation
and the dtype polarity -- were confirmed correct and are unchanged. What follows
repairs one false citation and five gate gaps.

**The false citation.** The header claimed that `gguf-py/gguf/tensor_mapping.py`
maps `vision.blocks.{bid}.attn.wqkv` to V_ENC_ATTN_QKV, "which the shared mmproj
base then writes as three SEPARATE `attn_q` / `attn_k` / `attn_v` tensors". That
mechanism does not exist at release `b10766`. `gguf-py/gguf/constants.py` spells
V_ENC_ATTN_QKV `v.blk.{bid}.attn_qkv`, `conversion/base.py` contains no
occurrence of `qkv` at all, the only converter that splits a fused vision qkv is
the model-specific `conversion/qwenvl.py`, and
`conversion/deepseek.py::DeepseekV4FlashVisionModel.modify_tensors` splits
`mlp.w1` only. Each of the four was re-read from the pin's own bytes over the
GitHub raw endpoint before the repair, rather than relayed from the review.

So a projector converted by the pinned oracle's own `convert_hf_to_gguf.py`
carries `v.blk.N.attn_qkv.{weight,bias}` -- 299 tensors at depth 32 against the
shipped file's 427 -- and this reader cannot load it. The fused arm is NOT
implemented. `RefuseUnsupportedDeepSeekV4ClipMmproj` refuses it by name, points
at issue #2411, and states that the file is not at fault, before
`RefuseUnaccountedDeepSeekV4ClipMmproj` can report that the artifact carries
tensors this build never reads. `## Owed` records the arm, the owning row and
the issue.

**The gate gaps.** `attn_out.weight` and `attn_out.bias` had no value case at
all: the fixture wrote them with their own exponent families and never read
either back, so 32 x 1M parameters were unmeasured. The aligner was checked at
flat index 0 only, and index 0 is the one element a transpose leaves alone, so a
row/column confusion in the square `mm.2` -- [4096, 4096] on the real artifact,
the one linear where a shape check cannot help -- was invisible. The shape guard
itself had no case. The absent-`clip.use_silu` branch had none either, although
`Options::emit_use_silu` already existed for it. And nothing bounded the
geometry read from `clip.*` before it became a `resize` argument.

**Mutation evidence.** Each mutation was applied to the production source alone,
`clip_mmproj_gguf.cpp.o` was confirmed to rebuild, the suite was run, and the
file was restored and verified with `sha256sum -c` against
`ee7b510e6a9eea39a57d8dcab95a7cadfac10eba4e069ae55f07b26acc0feed6`.

| Mutation | Case reddened | Failed assertions |
|---|---|---|
| source `out_weight` from `attn_q.weight` | the attention output projection | 128 |
| source `out_bias` from `attn_q.bias` | the attention output projection | 16 |
| transpose the square `mm.2` | the aligner and the sentinels | 132 |
| delete the `Require` shape check | a wrong-shaped tensor names both shapes | 3 |
| accept an absent `clip.use_silu` | a projector that declares no `clip.use_silu` | 1 |

The transpose figure is the measurement, not a round number: 132 is 144 elements
less the 12 on the diagonal, which is exactly the set a transpose can move. The
old index-0 check would have reddened on none of them.

The fused-layout and out-of-range-`block_count` cases needed no mutation,
because the code they gate did not exist. They started red together: 18 cases,
16 passed, 2 failed, 2198 assertions with 7 failed. The fused case failed on
"NOT IMPLEMENTED", on "2411" and on the absence of "NEVER reads"; the geometry
case failed on all four of its message assertions.

**A red-first case for an unbounded allocation performs the allocation.** The
first draft of the geometry case used `block_count = 4000000000`, which is what
the defect admits. With no guard in place that value reached
`blocks.resize(static_cast<size_t>(config.depth))` and asked for about 80 GB. It
tripped the GLOBAL Linux OOM killer twice on this box -- "Out of memory: Killed
process (test_deepseek_v) anon-rss:80197996kB" -- and took unrelated processes
with it. The case now asserts on the PARSED VALUE: `4096` is absurd
for a tower the artifact ships at depth 32, it is refused by name, and without
the guard it allocates a few megabytes and fails on the message. A test whose
only failure mode is `bad_alloc` is a crash, not a gate. Every test run in this
repair was made under `ulimit -v 6000000`.

**Five of the seven geometry bounds were held by no test, and the W3A evidence
did not distinguish the code from the gate.** SUPERSEDES the counts below and
`750cc6626`'s account of the geometry guard. `DeepSeekV4ClipMmprojVisionConfig`
calls `RequireGeometry` on seven `clip.vision.*` fields, and only `block_count`
and `embedding_length` had a case. A fresh review deleted the bounds on
`head_count`, `feed_forward_length`, `projection_dim`, `projector.scale_factor`
and `patch_size` and the suite stayed green; reproduced here by deleting all five
at once, which left 18 cases and 2198 assertions passing. Two of the five are
worse than a tower that runs and is wrong: `projector.scale_factor` at 0 divides
by zero in `aligned_rows`, and `head_count` at 0 divides by zero in `head_dim`.

Each of the five now has a case at both ends -- 0, which is absent-in-effect, and
`1 << 21`, which is above `kMaxGeometry` -- and each bound is held individually.
Deleting any ONE of the five reds 4 assertions in
`every clip.* geometry key is bounded BY NAME`; the file was restored and
verified against `ee7b510e6a9eea39a57d8dcab95a7cadfac10eba4e069ae55f07b26acc0feed6`
after each. The `Options` override moved from one field per key to a map keyed by
the key itself, so the seventh field and any future one costs a map entry rather
than a struct field. `test_deepseek_v4_mmproj` now reports 19 cases and 2218
assertions.

**After.** `test_deepseek_v4_mmproj` reports 18 cases and 2198 assertions, up
from 13 and 999. `test_clip_mmproj_gguf` reports 9 cases and 272 assertions,
unchanged, because the Qwen3-VL arm is deliberately untouched. `ctest
--test-dir build-repair3a -R 'deepseek_v4_mmproj|clip_mmproj_gguf'` passes 2/2
on a Release CPU build with `-DVLLM_CPP_CUDA=OFF`.

The reader is still not reached from production, and `## Owed` still names W4 as
the owner of the wiring. Three further gaps are recorded there and not fixed:
the unkeyed vision `rope_theta`, the four `clip.vision.image_*` preprocessor
keys, and the `general.alignment` fallback the shared fixture cannot yet
exercise.
