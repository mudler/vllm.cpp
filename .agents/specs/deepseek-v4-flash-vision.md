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
- **THE OFFICIAL SAFETENSORS VISION ARM LOADS, AND ITS REAL PAYLOAD HAS NEVER
  BEEN READ.** `src/vllm/model_executor/models/deepseek_v4_vision_weights.cpp`
  materializes the released 267-tensor BF16 vision group out of the checkpoint's
  own shards and `LoadDeepseekV4ForCausalLM`'s safetensors branch now attaches
  the tower, which closes the "MATERIALISING them is owed" note that branch
  carried in prose. EVERY GATE OVER IT IS SYNTHETIC. The pinned
  `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at
  `86f746b36186f0e567729a5c06a8c918caba82a9` is 156.287 GiB over 48 shards, it
  is staged on no device this row can reach, and nobody has authority to fetch
  it here, so not one weight byte of the official artifact has been read by this
  tree. What the arm is held to instead is the artifact's own METADATA: the
  released `config.json` and the shard-1 safetensors HEADER, committed under
  `tests/parity/goldens/deepseek_v4_vision/` and rebuilt from the pinned
  revision by `scripts/check-deepseek-v4-vision-manifests.py --refresh`, which
  reads two HTTP ranges and no payload. So the tensor NAMES, SHAPES, DTYPES and
  COUNTS are pinned to the real file, and the VALUES the loader produces are
  proven only on a synthetic fixture built to that header. A first real load,
  and any oracle or device gate for this arm, are owed by issue #2411 and the
  multi-device official-arm gate this spec's "Released artifact and geometry"
  section already scopes to tensor parallelism 4.
- **BOTH FILES ABOVE ARE PORTS, and the line they came from is preserved.** The
  loader and the manifest checker were ported from a parallel implementation of
  this row at `row/MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm-CODEX-LINE`
  (`3f3860851`), which forked before this spec's amendment and built against
  different vision types and a single combined-GGUF vehicle. Each file's header
  records what changed and why. Two of those changes are behavioural rather than
  stylistic and are recorded here because a reviewer may want to revisit them:
  the loader COPIES the group into owned storage instead of borrowing the
  safetensors mmap, because a safetensors payload offset carries no alignment
  guarantee and a borrowed bf16 view can begin at an odd address; and the
  checker's DEFAULT MODE now verifies the committed fixtures offline, because a
  record gate that must reach `huggingface.co` cannot run in CI. The ported
  line's combined-GGUF vision entry point was deliberately NOT taken: this row's
  vehicle is the two-file llama.cpp one, which `clip_mmproj_gguf.cpp` reads.
- **The PAGED attention arms cannot express the image-span exemption, and they
  REFUSE it.** `vt::AttentionWindow` carries one window per call, so the mask is
  per-call while the exemption is per-position. With `sliding_window = 128` and a
  384-token block, clipping the span away leaves two thirds of it invisible and
  the argmax plausible. `AttentionBlock` therefore refuses a step that carries an
  image span on a windowed paged layer, by name, and
  `test_deepseek_v4_mm_reach` drives that refusal through
  `ModelRegistry::Forward`. THIS IS THE ARM A REAL ENGINE TAKES, because
  DeepSeek-V4 publishes a multi-cache topology, so the image path is served on
  the non-paged branch only until the per-position mask lands. Issue #2411 and
  row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` own it.
- **The two DEVICE routers' media refusal was NEVER DRIVEN, and stays
  unmeasured.** W7-CUDA built and ran this architecture's suites on a CUDA queue
  on `thor:gpu0`, and no case reaches `DispRoute` with image rows on the
  `be.device` or glue arm, so the `VT_CHECK` below has still never executed. The
  device run happened; this particular refusal was not exercised by it, which is
  a different thing and is recorded rather than glossed. A case that drives an
  image step onto a device router is owed by issue #2411 and W7-CUDA.
- **The two DEVICE routers take one bias pointer per call.** `DispRoute` refuses
  a step carrying image rows on the `be.device` and glue arms rather than routing
  them on the text bias, and the two resident single-token decode arms refuse an
  out-of-vocabulary identifier. Owed by issue #2411 and W7-CUDA.
- **The DEVICE decode attention kernel takes no per-key index list, and W4 made
  that a divergence.** `deepseek_v4.cpp`'s `dev_attn` arm calls
  `DsaDevice()->decode_attn`, which derives its own key range from `kv_base + t`
  and attends the whole causal prefix. Until W4 that was the same list the host
  arm built. It is not any more: W4 made `sel` windowed and span-aware, so on a
  layer with no compressor at the released `sliding_window = 128` the host arm
  attends 128 rows while the kernel attends every one, and inside an image span
  the host arm attends forward while the kernel does not. The arm's own comment
  asserted the opposite and has been corrected. Neither existing refusal covers
  it -- `dev_attn` is independent of `be.device` and of `GlueDev`, so
  `DispRoute`'s media refusal does not reach it, and `paged_attn` is false in
  that branch -- so both conditions are now REFUSED BY NAME there.
  **THE WINDOWED HALF IS NOW MEASURED, W7-CUDA.** On `thor:gpu0` with
  `VT_V4_DEVICE_ATTN=1`, a CUDA queue and the V4 device kernels all live,
  `deepseek_v4.cpp:1319` threw by name at `sliding_window 128` — rc job
  `665b2427-4b85-4e75-916b-d3ad3345ea24`, and see `### W7-CUDA evidence`. The
  refusal is necessary and it is kept.
  **THE IMAGE-SPAN HALF IS STILL UNMEASURED**, and this sentence is the record
  of that: nothing in the suite drives an image span through `dev_attn`, so the
  second `VT_CHECK` has never executed. The windowed and span-aware device
  kernel, and a case that drives an image span onto this arm, are owed by issue
  #2411 and W7-CUDA.
- `ResidentWeight`'s device-staging arm in
  `include/vllm/model_executor/models/dense_attn_block.h` drops `q8_0_aligned`
  and `repacked` while guarding `elem_kn_repacked`, so the shared seam cannot
  carry a device-relevant storage layout for any model that inherits it. Found
  while repairing this row's own private copy of the same defect. It is a
  shared-seam gap outside this row, and
  [#2992](https://github.com/mudler/vllm.cpp/issues/2992) owns it.

- `V4GgufCtx::Vec` is still GEOMETRY-BLIND for four more per-layer vectors in
  the GGUF arm: `ffn_gate_tid2eid.weight`, `hc_ffn_base.weight`,
  `hc_ffn_fn.weight` and `hc_ffn_scale.weight`. `Vec` validates residency and
  role and no shape at all, which is exactly the F2 weakness `Vec1D` closed for
  the two router biases. W3B correctly did not chase them, since each has its
  own expected width and the safetensors arm already gets one from
  `carried.Float(..., {ne})`; nothing recorded them either, and this does.
  Issue #2411 and W4 own giving each its declared width.

- `608f403a3` changes product code in the W2 tower, which nothing reaches, and
  its body does not carry the `AGENTS.md` §"Nothing lands dead" declaration. The
  substance is met -- the subject carries the row ID and the issue, and this
  section covers the whole tower -- so this is a record note and not a rewrite of
  history. It is the same omission `435942c0d` was written to close for the
  commit before it.

- The first TP4 oracle run and committed evidence are owed by issue #2411 and W1.
- **CLOSED BY W6 for the TEXT half, OPEN for the served path.** The unsloth
  GGUF arm's first load and generation ran on `thor:gpu0`, rc job
  `a71c6002-4663-4efe-9da8-cda87e6bc4ff`: `deepseek-v4-gen` loaded the pinned
  `UD-IQ1_S` shards (`layers=43 experts=256 vocab=129280`) and generated
  greedily from a text prompt. The output is plausible but NOT oracle-gated,
  and no speed claim is made. See `### W6 evidence`. No request of either kind
  can be SERVED yet; the entry below names the refusal and its owner.
- **CLOSED BY W6.** The first `llama-cpp-dsv4vision` build and run.
  `llama-mtmd-cli` from release `b10766`, built CPU-only on `thor:gpu0`, loaded
  the 82 GB `UD-IQ1_S` language model and `mmproj-BF16.gguf` together, encoded
  an image and generated a description of it, rc job
  `b69b2fb9-23b9-42b8-b755-62b8ee93b6ea`. The oracle file's `gateable` value
  follows the parity verdict in `### W6 evidence`.
- **NO DEEPSEEK-V4 REQUEST CAN BE SERVED, text or image, until
  `KV-DSV4-MULTICACHE` W5 lands (#2455).** `vllm-server --model <shard1>
  --mmproj <mmproj>` dies at engine start with `server: fatal: vt: cache_dtype:
  an MLA KV cache has its own quantized page formula upstream (fp8_ds_mla,
  kv_cache_interface.py:398-410). W1 landed that page formula but no fp8_ds_mla
  store or read, so a page sized for it would hold bytes nothing writes.`
  DeepSeek-V4's own KV factory publishes `fp8_ds_mla` specs, and no store or
  read for that format exists. This is not this row's to fix: row
  `KV-DSV4-MULTICACHE` W5 and issue #2455 own it. Measured by rc job
  `a71c6002-4663-4efe-9da8-cda87e6bc4ff`, step C.
- `exp_probs_b_vl` is ACCOUNTED FOR in all three loader arms by W3B and LOADED
  in the two that materialize a tower, the GGUF arm and the EXL3 carried arm.
  The official dense safetensors arm accounts without materializing, exactly as
  it does for every other tensor, so its W2b residual covers this one too.
  **ALL THREE ARE CLOSED BY W4.** `SqrtSoftplusRouteTopk` selects between the
  two biases per token, an image row on a hash layer takes the vision bias in
  place of `tid2eid`, and `DeepseekV4VisibleRows` carries the non-causal
  image-span rule. The remaining device-arm gaps are listed below.

  **WHAT THE ORACLE DOES, and where our intent differs from it.** An earlier
  wording of this entry described the first two as "per-token selection" and as
  "an image row takes `exp_probs_b_vl` while a text row takes `tid2eid`", and
  attributed that shape to the oracle. It is not the oracle's shape. In
  `llama_model_deepseek4::graph::graph` in `src/models/deepseek4.cpp` at
  `llama-cpp-dsv4vision`, which is release `b10766` -- the merge commit
  `9400c8946e4da5e7694f2c26d6d4e50e14b690fa` of "model: correctly support input
  vision for deepseek4 (#28154)" -- the selection is PER
  UBATCH: `const bool is_media = ubatch.embd != nullptr;` and, when it is set,
  every layer takes `ffn_exp_probs_b_vl` if the layer has one and the
  `il < hparams.dsv4_hash_layer_count` branch is SKIPPED ENTIRELY, so
  `ffn_gate_tid2eid` is never consulted. The image-row/text-row split on one
  batch does not happen there, because a media ubatch carries no text rows.

  **W4 CHOSE PER TOKEN, and this is the argument.** Three grounds, in order of
  weight:

  1. It AGREES with the oracle on every input the oracle can express. A media
     ubatch carries no text rows, so "every row is media" and "this row is
     media" select identically at `llama-cpp-dsv4vision`. The divergence is
     therefore an EXTENSION to inputs llama.cpp cannot build, not a
     contradiction of it.
  2. Our step is not a ubatch. This engine batches continuously, and one step
     mixes an image request's prefill rows with other requests' decode rows.
     `MultiModalForwardInput` is set for the whole step, so a whole-step flag
     would route another request's TEXT tokens on the vision bias -- which the
     oracle never does on any batch it can construct.
  3. The hash question has a per-row answer, and it is the SAME answer the
     oracle gives wholesale. A hash layer carries `exp_probs_b_vl` and no
     `exp_probs_b`; an image row has no identifier worth hashing, so it takes
     the vision bias and the learned top-k route, and a text row in the same
     step still hashes through `tid2eid`. The oracle skips the hash branch for
     the whole ubatch only because no text row is there to keep it.

  **A SECOND DIVERGENCE FROM THE ORACLE, RECORDED RATHER THAN CHANGED.** On a
  media batch whose layer carries NO `exp_probs_b_vl`, the oracle still skips the
  hash branch and takes plain unbiased top-k -- its selection is
  `layer.ffn_exp_probs_b_vl ? that : nullptr`, and a null bias is simply not
  added. `deepseek_v4.cpp` REFUSES that layer by name instead, because a layer
  that was handed an image row and has no vision bias is a TEXT checkpoint being
  asked to route an image, and routing it on the text bias or on no bias at all
  would be fluent and wrong. It is not live for the released 43-layer file,
  which carries the tensor on every layer, and it is the first thing to
  reconsider if a partially converted vision file has to load. Issue #2411 owns
  it.

  **AND W4'S VISIBILITY RULE IS STRICTER THAN THE ORACLE'S.** llama.cpp's
  `set_input_kq_mask_impl` exempts a key from the window when
  `p0 >= seq_pos_min[seq_id]`, with no upper bound; `DeepseekV4VisibleRows`
  bounds the exemption at `span_end`. The two agree on every input the oracle can
  build, because a media ubatch IS the span there and nothing follows it inside
  the batch. Ours is the narrower rule on a mixed step, which is the same
  argument the per-token bias choice rests on: this engine batches continuously
  and one step can carry rows after the span that must stay causal. Recorded so
  the difference is a decision rather than a discovery.

  WHICH ROWS ARE IMAGE ROWS is read from the step's own identifiers. The
  processor writes `vocab_size + DeepSeekV4ImageTokenType` at every position of
  an image block, so `MoeBlock` needs no new forward channel and a text step,
  whose identifiers are all below the vocabulary, is byte-identical.

  The two DEVICE routers take one bias pointer per call and have no per-row
  selector, so `DispRoute` REFUSES a step carrying image rows on those arms by
  name rather than routing them on the text bias. The kernel change is owed by
  issue #2411 and W7-CUDA. The two resident single-token decode arms refuse an
  out-of-vocabulary identifier for the same reason, and they also read `embed`
  with no bound, which that refusal now closes.

  **The refusal ORDER, CLOSED BY W4.** `RefuseDeepSeekV4ClipMmprojArm` holds it
  in one function and `model_loader.cpp` calls that function rather than its
  parts, so a second call site cannot get it wrong. The order is gated at the
  production call site: `test_deepseek_v4_mm_reach` drives
  `LoadedEngine::FromModelDir` with a FUSED-qkv projector -- the layout the
  pinned `convert_hf_to_gguf.py` actually emits -- and asserts the message names
  `attn_qkv` and issue #2411 rather than blaming the file for carrying tensors
  the reader never reads. Swapping the two calls reddens it.
- `scripts/check-dsv4-gguf-namemap.py` is owed the vision manifest. It generates
  1328 expected names and asserts exact set-equality against the TEXT artifact,
  so the 1371-name vision artifact fails it by construction and no gate covers
  the shipped vehicle's language half. `scripts/dsv4_vision_gguf_manifest_names.txt`
  is the committed fixture and the measurement above is the red-before input.
  The checker change is not made here because it is a semantic checker change
  and needs its own red-before evidence. Issue #2411 and W3 own it.
- **CUDA: the VISION half is CLOSED by W7-CUDA; ROCm and Vulkan are still
  owed** by #2411 W7-ROCM and W7-VULKAN. Every run uses `rc`; a CPU fallback is
  not evidence for any of them. The vision tower now runs on a CUDA queue on
  `thor:gpu0` (sm_110) against the real projector and matches llama.cpp
  `b10766` inside W6's declared bound; see `### W7-CUDA evidence`. What CUDA
  still cannot do is listed in the four entries below.

- **`vt: MatVec weight size mismatch` IS NOW THE FIRST BLOCKER FOR A SERVED
  IMAGE ON CUDA. It SUPERSEDES vision residency, which W7-CUDA fixed.**

  **THE BLOCKER ORDER ON THIS ROW HAS MOVED THREE TIMES UNDER MEASUREMENT, and
  each move was only visible because the previous blocker was genuinely
  repaired.** A reader needs to know which are closed and which is live:

  | # | Blocker | State |
  |---|---|---|
  | 1 | `fp8_ds_mla` KV cache at engine start | **NOT what stops a CUDA build.** W6 measured it on a CPU build; `KV-DSV4-MULTICACHE` W5 (#2455) owns it and it is untouched here |
  | 2 | `DeepSeek-V4 vision queue and weights must share one device` | **CLOSED by W7-CUDA.** See `### W7-CUDA evidence` |
  | 3 | `vt: MatVec weight size mismatch at deepseek_v4.cpp:504` | **LIVE. This entry.** |

  **THE EXACT FAILING INVOCATION.** `test_deepseek_v4_mm_chat`'s served image
  request dies with
  `engine-fatal: EngineCore busy loop threw: vt: MatVec weight size mismatch at
  deepseek_v4.cpp:504`. `MatVec` has exactly ONE call site in that file, `:567`,
  inside `Gemm`'s HOST-FLOAT FALLBACK:
  `const std::vector<float> y = MatVec(wf32, &x[t * K], N, K);` — so `out = N`,
  `in = K`, and the guard that fires is
  `VT_CHECK(static_cast<int64_t>(w.size()) == out * in, ...)` at `:504`.

  **IT IS NOT THE DEVICE GEMM PATH**, and calling it one would be wrong. `Gemm`
  takes its keep-quant arm only when
  `be.gguf != nullptr && wq != nullptr && !wq->Empty()`, and otherwise falls
  through to that host loop REGARDLESS of device. The failing code is host code.
  What is device-specific is its REACHABILITY: on a CPU build the request never
  arrives, because `ForwardDevice` refuses first at
  `VT_CHECK(deepseek_v4::V4DeviceKernelsAvailable(), kDevicePending)`, so only a
  build with the V4 device kernels can get this far.

  **THE ASYMMETRY IS THE FINDING.** The keep-quant arm carries a NAMED shape
  refusal (`keep-quant GEMM: weight shape mismatch: want [N=..,K=..] got [..]`)
  while this fallback arm's guard is ANONYMOUS. The same wrong shape is
  diagnosable on one arm and nameless on the other. The tree already says what
  that costs: `deepseek_v4_weights.cpp:346` records that the assertion is
  "unconditional (a plain `VT_CHECK` and not an `assert`, so it survives
  `NDEBUG`)" and that the throw "names neither the tensor, nor the layer, nor
  the geometry, nor what is missing".

  **WHAT IS UNMEASURED, and is not guessed here:** the `N` and `K` values, which
  tensor, and which layer. This throw names none of them by construction, so
  recovering them needs an instrumented device run. It is NOT the aarch64 repack
  path — the failure is byte-identical with `VT_CPU_QUANT_REPACK=0`.

  **A STALE CROSS-REFERENCE a reader will otherwise chase, and it is THREE
  places rather than four.** `deepseek_v4.cpp:728` and `:734`,
  `deepseek_v4.cpp:832` and `deepseek_v4_weights.cpp:344` and `:347` cited this
  throw as `deepseek_v4.cpp:413`. `deepseek_v4_weights.cpp:1068` names the same
  anonymous message and carries NO line number, so it was never stale; this
  record said four and the tree says three. The guard sits at `:504`, which is
  what the measured failure reports, and the three stale citations are corrected
  to `:504` here. A fourth `:413` citation lives in
  `tests/vllm/models/test_deepseek_v4_exl3_forward.cpp:443,446`, which belongs to
  `MODEL-DSV4-EXL3` and is left to that row.

  Root-causing it, and giving the fallback arm a named refusal, are owed by
  issue #2411 and W7-CUDA, and by the row-owned local issue this repair filed
  for it under `.agents/issues/MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm/`.

- **`test_deepseek_v4_mm_chat`'s image branch encodes a CPU-ONLY PREMISE and
  fails on any CUDA build.** Its else-branch asserts the served error names
  `W7-device`, which is `kDevicePending` — and `ForwardDevice` guards that with
  `VT_CHECK(V4DeviceKernelsAvailable(), kDevicePending)`, a predicate that is
  FALSE exactly when the device kernels are absent. On a CUDA build the refusal
  therefore cannot fire, and the assertion can never hold. One assertion of 650
  fails for this reason (the sibling `deepseek_v4.cpp` check now passes, because
  the new message names that file). The case needs a device-aware expectation
  rather than a CPU-shaped one; owed by issue #2411 and W7-CUDA, and by its own
  row-owned local issue under
  `.agents/issues/MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm/`.

- **8 of 20 `test_deepseek_v4_mm_reach` cases FAIL ON AARCH64, and the cause is
  the i8mm quant repack rather than the device.** Every one throws
  `deepseek-v4 keep-quant expert/group slice requires non-repacked blocks
  (disable VT_CPU_QUANT_REPACK for the stacked-expert weights)` at
  `deepseek_v4.cpp:583`. **PROVEN by an A/B on the same binary and the same
  box**, not inferred from the message: with the repack ON the suite reads
  `20 | 12 passed | 8 failed`; with `VT_CPU_QUANT_REPACK=0` it reads
  `20 | 20 passed | 0 failed`. `vt::cpu::QuantRepackActive()` is true only on an
  aarch64 i8mm host, which is why these cases are green on the x86-64 devbox and
  red on `thor`. The row's gate therefore cannot run clean on an aarch64 host
  without that flag. Owed by issue #2411, and by its own row-owned local issue
  under `.agents/issues/MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm/`.

- **`test_serve_deepseek_v4_mm` TIMES OUT at 1800 s on a CUDA build, and why is
  UNKNOWN.** It produced no output before CTest killed it, on both the red and
  the green run, so nothing here attributes it. It is not asserted to be related
  to the vision path. Owed by issue #2411 and W7-CUDA, and by its own row-owned
  local issue under
  `.agents/issues/MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm/`.
- **CLOSED BY W4.** The W2 vision tower and aligner were unreachable from a
  production entry point, and are not any more. `DeepseekV4LoadedModel::
  vision_tower` builds `DeepSeekV4Vision` and `EncodeMmDeepseekV4ForCausalLM`
  runs its `Forward`, reached from `ModelRegistry::EncodeMm`.
  `DeepSeekV4VisionCapture` and `DeepSeekV4VisionRopeCosSin` stay test-only, and
  deliberately: the first is a parity-gate tap and the second is a host oracle
  for one, so neither is a capability a user arrives at. `VisionForward` and
  `AlignerForward` are reached through `Forward`, which composes them.
- **CLOSED BY W5, except one function.** W1's request path is reached.
  `MakeDeepSeekV4ChatSeam` is registered for `DeepseekV4ForCausalLM` in the
  per-architecture multimodal chat registry, so `InstallMultiModalChatSeam` --
  the ONE production caller of `set_multimodal_chat_fn`, reached from
  `server_main.cpp` and now from `vllm_chat` -- builds it, and its chat function
  calls `EncodeDeepSeekV4Messages`, `DeepSeekV4ImageProcessor::ProcessImage`,
  `DeepSeekV4ImageProcessor::HashImage` and `PrepareDeepSeekV4Inputs` on every
  image request. `BuildDeepSeekV4ImageBlock` was already reached by W4.

  `ParseDeepSeekV4TaggedText` is STILL UNREACHED, and deliberately. It converts
  the compact `<image>path</image>` syntax into content blocks, where `path` is
  a FILESYSTEM PATH the encoder would then be asked to open. Wiring that into a
  chat body would let a request name a local file, which is a different feature
  with a different threat model from an inline `data:` URI, and this wave did
  not add it. Owed by issue #2411 and row
  `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`.

- **THE MULTIMODAL CHAT PATH IGNORES `--chat-template` ON THIS ARCHITECTURE.**
  The other two registered seams inject a marker string and render through
  `ctx.prompt_fn`, the server's Jinja template. This one calls the pinned
  `encode_messages` port instead, because `encoding_dsv4.py` is where this
  model's prompt is defined and its image handling is inseparable from the rest
  of it: the placeholder replaces the content block in place, a text block that
  already carries the placeholder is refused there, tool results are sorted and
  merged around it, and the thinking-mode elision decides which turns survive to
  carry it. The TEXT path still renders through the server's template, so a
  conversation carrying both kinds of turn can be templated two ways. Reconciling
  them is owed by issue #2411.

- **THE PNG/JPEG CODEC AND THE `http(s)` FETCH ARE STILL NOT IMPLEMENTED**, and
  W5 refused them rather than vendoring a decoder. The codec is the LIBRARY's --
  `oai::DefaultImageCodec`, consumed by three architectures and now by the C ABI
  -- so implementing it inside a model row would land a cross-model capability
  under a model row. What W5 did change is the STATUS a user meets on the
  DeepSeek path: `DefaultImageCodec` and `DecodeDataUri` throw
  `std::runtime_error`, which `api_server.cpp:373` maps to HTTP 500
  "InternalServerError", so a `data:image/png;base64,...` body read as a server
  fault. The DeepSeek seam re-throws them as `InputValidationError`, which maps
  to 400 with each residual's own message intact. **The Qwen3-VL and dots3-note
  seams still answer 500 for the same body**, which is a defect this wave found
  and did not widen its scope to fix; it needs an issue of its own and is owed
  by issue #2411 until one exists.

- **IMAGE PREFILL IS NOT ATOMIC AT THE SCHEDULER, and the step is refused
  instead.** The spec's data flow requires an image span to fall inside one
  prefill chunk. `Scheduler::try_schedule_encoder_inputs` can do that -- it
  rolls a step back to before an item when
  `SchedulerConfig::disable_chunked_mm_input` is set -- but that flag defaults
  to false and NOTHING in this tree can turn it on: no command-line flag, no
  `include/vllm.h` field, and no per-architecture channel through which a model
  could ask for it. Adding one is a shared scheduler-policy seam rather than a
  model change. Until it lands, `DeepseekV4ImageSpans` refuses by name any step
  whose media rows are not all inside complete blocks, which W5 extended to the
  INTERIOR chunk (see the W5 evidence below). Owed by issue #2411 and row
  `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`.

- **THE SERVED PATH CANNOT GENERATE ON A CPU BUILD, and a TEXT request on the
  synthetic fixture is UNSTABLE.** Two separate facts, measured together while
  gating W5's server surface.

  `DeepseekV4Model::ForwardDevice` is what the runner's gather-logits path
  reaches for EVERY request on this architecture, and a CPU build carries no V4
  device kernels, so every served request -- text or image -- is refused by name
  at `deepseek_v4.cpp:4345`. That refusal is W5's own reachability evidence at
  the server surface, because nothing short of the registered forward can
  produce it, and serving this architecture on a device is W7-CUDA's.

  Separately, a request on the synthetic `deepseek4` GGUF is unstable: the same
  binary segfaulted in `InputBatch::add_request` on three of six runs and
  otherwise died in `GPUModelRunner::gather_block_table`, at a one-token prompt
  as readily as at a 260-token one, and the case does not gate on the text
  prompts. Both signatures are issue
  [#3027](https://github.com/mudler/vllm.cpp/issues/3027).

  **The `gather_block_table` signature is root-caused and repaired**, in the
  W4/W5 reconciliation. `full_attn_group_id_` is a -1 sentinel meaning "this
  model published no `kFullAttention` or `kMlaAttention` group", which is TRUE
  OF THIS ARCHITECTURE ON EVERY REQUEST, and the full-attention gather passed it
  straight to `MultiGroupBlockTable::operator[]`, which casts its index to
  `size_t`. Every served DeepSeek-V4 step therefore read a `BlockTable` object
  that does not exist, and the `max_num_blocks_per_req` it found decided the
  step: a garbage zero gathered an empty table and the request went on to the
  forward, a garbage negative made `num_reqs * cols` a ~1.8e19-element
  allocation and the engine's busy loop died with `std::length_error`. The
  outcome moved with the BINARY'S LAYOUT rather than with the request, which is
  why the W5 measurement read eight of eight and the merged branch read zero of
  five: merging W4 flipped it, and so did running one earlier case of the suite
  first. `gather_block_table` now answers an out-of-range group with an empty
  table, which is what `MakeCommonAttentionMetadata` is already written against
  for the same sentinel, and is byte-neutral for every model that publishes a
  full-attention group. Whether this architecture's group should be CLASSIFIED
  as the target attention group is a different question, owed by row
  `KV-DSV4-MULTICACHE` W3 (#2068).

  **The `InputBatch::add_request` signature is not explained and not repaired.**
  #3027 stays open for it. It also means W5's "the served image request reaches
  `ModelRegistry::Forward`" evidence rested on an out-of-bounds read returning a
  convenient zero; the claim itself survives, and is now deterministic, but it
  was not measured until this repair.

  Two further engine conditions had to be pinned for the fixture to load at all,
  and each is a gap rather than a preference: the file carries no
  `deepseek4.context_length`, so the engine resolves `max_model_len = 0` and
  `InputBatch`'s per-request token row has no width (a SIGSEGV, not an error);
  and prefix caching must be off, because this architecture's KV topology gives
  the block pool a hash-block size that differs from its block size and
  `BlockPool::cache_full_blocks` refuses that pair by name. All of it is owed by
  issue #2411.
- **CLOSED BY W4.** The W3A `deepseek4v` mmproj reader is reached.
  `src/vllm/entrypoints/model_loader.cpp` branches on `clip.projector_type` and
  calls `RefuseDeepSeekV4ClipMmprojArm` before the tokenizer, and
  `LoadDeepseekV4ForCausalLM` calls `LoadDeepSeekV4ClipMmprojArm` through
  `ModelSource::mmproj`. The Qwen3-VL discriminator still refuses `deepseek4v`
  and the W3A gate still asserts that it does; the branch is what stops it being
  reached. The four sentinel vectors are consumed by
  `EncodeMmDeepseekV4ForCausalLM`, which places one under each marker token of
  the image block.
- **CLOSED BY W6.** The pinned 934,462,656-byte `mmproj-BF16.gguf` has now
  been read AND RUN by this code. W3A gated the name map, the metadata map and
  the four layout joins against a synthetic fixture only. W6 loads the real file
  through `LoadDeepseekV4VisionRuntime`, runs the tower through
  `ModelRegistry::EncodeMm`, and compares the block with llama.cpp `b10766` on a
  real image; see `### W6 evidence`.
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
- **The windowed `dev_attn` refusal is labelled MEASURED, and NOTHING RE-CHECKS
  IT.** The label is history from one rc lease run on `thor:gpu0`, job
  `665b2427-4b85-4e75-916b-d3ad3345ea24`. No committed test sets
  `VT_V4_DEVICE_ATTN`, so no gate in this tree drives that refusal: it is
  CUDA-only and a CPU build cannot reach it at all. A regression that deleted or
  weakened the refusal would leave every gate green and would be visible only on
  the next manual lease run. A case that sets `VT_V4_DEVICE_ATTN` on a CUDA
  build is owed by issue #2411 and W7-CUDA.
- **NO COMMITTED GATE PROTECTS THE W7-CUDA STAGING FIX**, and this is the
  measured statement of it rather than an estimate. Making
  `DeepSeekV4Vision::Impl::EnsureResident` a no-op in a scratch copy leaves the
  whole CPU DeepSeek-V4 family gate GREEN, because `EnsureResident` returns on
  its first line for a CPU queue with host weights and every CPU case is in
  exactly that state. The two cases that do measure the staging are in
  `test_cuda_deepseek_v4.cpp` and need a CUDA queue plus the V4 device kernels,
  so on any CPU host they return early and the suite exits 77. What this wave's
  green covers is therefore the CPU arm's unchanged behaviour; the staging
  itself is covered only by a lease run. Issue #2411 and W7-CUDA own a gate that
  runs on a leased device.
- **`tools/parity/dsv4v_w6_compare.py` CONTAINS NO BOUND AND EMITS NO VERDICT.**
  It prints and writes statistics — `mean_rel_l2`, `mean_cos`, `min_cos`,
  sentinel exactness, the permutation summary — and returns 0 whenever the
  shapes match. The only `verdict` key it ever writes is `SHAPE_MISMATCH`. The
  `<= 4.9%` cells mean relative L2 and `>= 0.998` mean cosine judgement recorded
  in `### W6 evidence` and `### W7-CUDA evidence` is therefore PROSE ARITHMETIC
  performed by a reader against that output, not something the harness checks. A
  future run that drifted past the bound would still exit 0. Teaching the
  comparator its bound and a pass/fail verdict is owed by issue #2411.
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

### W7-CUDA evidence — the vision tower on the device, and what the device still cannot do

Every job ran on `thor:gpu0` through `rc`, and every artifact was written to
`/workspace/dsv4-vision/w7-out/` as well as to stdout, because `rc` logs age out
within a day. Thor is **sm_110**, outside the vendored FlashAttention-2 arch set,
and the configure log says so on every run:
`CUDA feature fa2: DISABLED (no requested arch in [110] provides it)`. Two jobs
carry the result:

| Job | Head | What it established |
|---|---|---|
| `14908980-7670-4283-a798-4247481f0bf2` | `4abe547d2` | THE RED. The tower could not run on CUDA at all |
| `665b2427-4b85-4e75-916b-d3ad3345ea24` | `4abe547d2` + the staging fix | THE GREEN, and the aarch64 attribution |
| `c472faab-347f-451d-865d-e844aff15e77` | (artifacts only) | the device block against the llama.cpp oracle directly |

**THE BUILD IS A CUDA sm_110 BUILD, and that is proven rather than assumed.**
CUDA **13.0.88**, installed by the job: the worker image carries no toolkit, and
the box's leftover system `nvcc` is **12.0**, which cannot target sm_110 at all.
41 `.cu.o` objects, and `cuobjdump --list-elf` over all 41 reports **41 sm_110**
with `objects scanned: 41`. `ldd` on `tests/test_cuda_deepseek_v4` resolves
`libcudart.so.13` and `libcublasLt.so.13`. The first run's `ldd` line asked for
`libvllm.so` and got "No such file or directory"; that was a defective proof
line, not a finding — `CMakeLists.txt:732` is `add_library(vllm STATIC ...)`, so
this tree has no shared object. It is repaired to target an executable.

**1. THE RED: the vision tower could not take a CUDA queue.** The `deepseek4v`
mmproj reader hands the tower HOST views — `clip_mmproj_gguf.cpp`'s `HostView`
says so in its own comment, "W4 owns the upload, so this wave keeps every weight
on the default device" — and W4 did the routing rather than the upload, so
nothing ever uploaded them. `DeepSeekV4Vision::ValidateQueue` then refused every
CUDA queue. Measured on the real 934,462,656-byte `mmproj-BF16.gguf`, all four
`lead_pad` rungs aborted:

```text
terminate called after throwing an instance of 'std::invalid_argument'
  what():  DeepSeek-V4 vision queue and weights must share one device
```

**The same sentence killed a SERVED image request**, which is what makes this a
capability gap and not a probe artifact: `test_deepseek_v4_mm_chat` died with
`engine-fatal: EngineCore busy loop threw: DeepSeek-V4 vision queue and weights
must share one device`. So vision residency, and NOT the `fp8_ds_mla` KV cache
(#2455), is the FIRST blocker a served image meets on a CUDA build.

**THE CONTROL THAT MAKES IT A DEVICE RESULT.** The same binary's CPU arm
reproduced W6's block BYTE FOR BYTE (`cmp`, reported as
`CPU_CONTROL_IDENTICAL`). The refusal is therefore a property of the device
path, not of this build.

**2. THE FIX, and it is the smallest one that reaches the capability.**
`DeepSeekV4Vision::Impl::EnsureResident(queue)` stages the tower to the queue's
device on first use and rebuilds the `MlpGateUpMethodBase` borrows against the
staged tensors, because `BorrowResidentWeight` aliases whatever device its
argument declares and leaving them alone would hand the shared seam a host
pointer labelled with a device. **ONLY HOST -> DEVICE IS ADDED.** A queue on one
device with weights already on a different one still hits the original refusal:
the refusal is NARROWED, never deleted, and a device case asserts that it still
fires.

**3. THE GREEN, against W6's own CPU block** (job `665b2427`, all four rungs
`RC=0`, `provider: cuda`):

| Check | Result |
|---|---|
| token count | 114, 115, 116, 117 for `lead_pad` 0-3 — the same as W6 |
| START, END, every NEWLINE, every PAD | **EXACT**, `f32_exact: true`, `max_abs 0.0`, every rung |
| row placement | the identity is the best cosine match for **100 of 100** rows, every rung |
| patch rows consumed | **IDENTICAL** (`max_abs 0.0`, `mean_cos 1.0`) — both arms read the same input |
| aligner cells | mean relative L2 **2.32%**, mean cosine **0.99970**, min cosine 0.99363 |
| vit, 784 rows | mean relative L2 **1.51%**, mean cosine 0.99979 |

**4. THE GREEN, against the llama.cpp `b10766` ORACLE DIRECTLY** (job
`c472faab`). Item 3 compares the device against OUR OWN CPU arm, which would
leave the oracle claim resting on two chained measurements. These are the
oracle's own dumps, captured by W6 from `llama.cpp` itself, and the numbers are
identical on all four rungs:

| Check, 100 image rows | CUDA vs oracle | CPU vs oracle (W6) |
|---|---|---|
| four sentinel kinds | **EXACT**, `max_abs 0.0` | EXACT |
| permutation | **identity best 100 of 100** | identity best 100 of 100 |
| cells mean relative L2 | **2.884%** | 3.83% |
| cells mean cosine | **0.99939** | 0.99899 |
| cells min cosine | 0.98653 | 0.96709 |
| vit mean relative L2 | **1.872%** | 2.45% |

**VERDICT AGAINST W6'S DECLARED BOUND, not against a number chosen here.** W6
set three conditions. (1) the four sentinel kinds are exact and every image row
is in its place — **met**, exactly, on every rung. (3) the shipped bf16 path is
no farther from the oracle than it is from its own f32 arm plus the oracle's own
floor, `<= 3.34% + 1.57% = 4.9%` cells mean relative L2 with mean cosine
`>= 0.998` — measured **2.884%** and **0.99939**, so **met**. Condition (2) is
about the f32 arm and no f32 device arm was run; it is untouched by this wave.
The device arm is CLOSER to the oracle than our own CPU arm is, and the residual
has W6's structure rather than a defect's: relative error tracks row norm
(`corr = -0.351`) while absolute error does not (`corr = +0.073`), and no
aligner row or column is loaded.

**5. THE DEVICE SUITES, and every skip named.** `test_cuda_deepseek_v4` ran
**29 cases, 0 skipped, 90082 assertions, all passed**, including the two cases
this wave adds. A grep for skip messages across the whole run returns NOTHING:
no case silently skipped. The suite exits 77 on a host with no CUDA, and that
was verified on the devbox, so its green here is a device green.

`ctest -R 'deepseek_v4|clip_mmproj_gguf'` reported **24 of 27 passed**. The
three failures are characterised below, and NONE of them is caused by this
wave's change, which before the fix touched only `tools/parity/`.

**6. THE THREE REFUSALS.**

- **The DEVICE decode attention refusal FIRES, and W4's "unmeasured" record is
  now measured.** With `VT_V4_DEVICE_ATTN=1` on a CUDA build at sm_110,
  `deepseek_v4.cpp:1319` threw by name: *"layer 0 runs the DEVICE decode kernel
  at sliding_window 128 ... Refused by name; the windowed device kernel is owed
  by issue #2411 ... Unset VT_V4_DEVICE_ATTN to take the host arm"*. It is
  NECESSARY and it is kept. **Its IMAGE-SPAN half did not fire**, because
  nothing in the suite drives an image span through `dev_attn`; that half stays
  UNMEASURED and `## Owed` says so.
- **The two DEVICE routers' media refusal was NOT driven.** No case reaches
  `DispRoute` with image rows on a device arm, so it stays unmeasured.
- **The paged image-span refusal** is gated on the host by
  `test_deepseek_v4_mm_reach`, and it is unchanged by the device: the predicate
  is `vt::AttentionWindow`'s one-window-per-call shape, which no device build
  alters.

**7. WHAT THIS DOES NOT SHOW.** One image at one size. No served image answer
exists yet (see `## Owed`). No speed was measured, and no speed claim is made.
No f32 device arm was run, so W6's condition (2) has no device counterpart. ROCm
and Vulkan are untouched.

### W6 evidence — the first real-weight run, and vision parity against llama.cpp `b10766`

Every job below ran on `thor:gpu0` through `rc`. Every result was also written to
`/workspace/dsv4-vision/w6-parity/` (steps A-C: `/workspace/dsv4-vision/w6-out/`)
on the NAS the workers see as `/workspace`. The artifacts are the pinned
`unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF` @
`b977d3c0ea2da58dbc12ddae8fb8951a7b3854d0`, sha256-verified:
`mmproj-BF16.gguf` (934,462,656 B) and the three `UD-IQ1_S` shards.

**A. PASS: the language model loads.** rc job
`a71c6002-4663-4efe-9da8-cda87e6bc4ff`, built from row head `4993c72b2`
(`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF`).
`deepseek-v4-gen --load-only --gpu` printed `LOADED: layers=43 experts=256
vocab=129280 has_gguf=1 | open 1.1s load 1508.4s | RSS 64.6 GiB`. The load
time is almost all first-touch paging off CIFS.

**B. PASS, NOT ORACLE-GATED: text generation.** Same job. `deepseek-v4-gen --gpu
--kv-cache --max-tokens 24 --prompt "The capital of France is"` generated
` Paris. The capital of France is Paris. The capital of France is Paris. ...`
greedily with no stop, ids `11111 16 455 6102 294 8760 344` repeating. `--gpu`
puts only the keep-quant GEMMs on CUDA; the rest is the CPU queue. "Paris" is
plausible and is not a token gate: no oracle ran this prompt. **No speed claim is
made**: the first step took 201.75 s of paging, and the numbers the tool printed
measure CIFS, not the engine.

**C. BLOCKED, owned elsewhere: the server.** Same job. `vllm-server --model
<shard1> --mmproj <mmproj>` exits at engine start with:

```text
server: fatal: vt: cache_dtype: an MLA KV cache has its own quantized page formula upstream (fp8_ds_mla, kv_cache_interface.py:398-410). W1 landed that page formula but no fp8_ds_mla store or read, so a page sized for it would hold bytes nothing writes. EITHER --kv-cache-dtype asked for a non-auto dtype, OR (DeepSeek-V4, #2455) the model's own KV factory published fp8_ds_mla specs and no flag was
```

No DeepSeek-V4 request can be served, text or image, until `KV-DSV4-MULTICACHE`
W5 (#2455) lands. `## Owed` records it.

**THE ORACLE BUILDS AND RUNS THE MODEL.** rc job
`b69b2fb9-23b9-42b8-b755-62b8ee93b6ea`. `ggml-org/llama.cpp` was cloned inside
the job and checked out at `9400c8946e4da5e7694f2c26d6d4e50e14b690fa`, and the
job asserted `rev-parse HEAD` against that value; `git describe` printed
`b10766`. It was built CPU-only and static: `cmake -G Ninja
-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF
-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF`, targets
`llama-mtmd-cli` and `dsv4v-oracle-dump`, `-j 4`. `llama-mtmd-cli -m <shard1>
--mmproj mmproj-BF16.gguf --image img392.png -p "Describe this image in one
sentence." -n 32 --temp 0 -c 4096 --no-mmproj-offload` loaded the 82 GB
language model and the projector, encoded the image in 71,076 ms, and answered
`This image is a colorful, abstract composition featuring a vibrant, swirling
background of concentric circles in hues of blue, green, purple, and pink,
overlaid with`. That fits the input: its blue channel is a radial sine, which
draws concentric rings.

**THE COMPARISON AND WHY IT MEASURES THE TOWER.** The image is 392x392 RGB,
generated deterministically by `tools/parity/dsv4v_w6_image.py` (rgb sha256
`fda46a17fe85ba9919956cb535fbcf5b61552b29cb6e5623df7031b55e0dff99`, png sha256
`3f4aeae0eed47f6f6bd898cadc42cc57ccb23ab86527508626e8057a168c8f9a`, the same
bytes). NEITHER SIDE RESAMPLES IT, and both halves of that claim were read at the
code, not assumed. 392 is a multiple of 14, and its area of 153,664 is above the
147,456 `image_min_pixels` the file carries, so both sides keep 392x392 as the
target. llama.cpp's `img_tool::resize` then COPIES when source and target sizes
are equal (`tools/mtmd/mtmd-image.cpp:50-54` at the pin). Our `ProcessImage`
transforms only when `height != best_height || width != best_width`. The oracle
logged `preprocess: 392x392 (from 392x392)`. The patch grid is 28x28, the
aligner grid 10x10, and the block `114 + lead_pad` tokens.

Our side is `tools/parity/dsv4v_w6_probe.cpp`, built from `4993c72b2` plus the
probe (not in the default build). It drives the SHIPPED path:
`LoadDeepseekV4VisionRuntime` on the real mmproj, `DeepSeekV4ImageProcessor` on
the raw RGB, `PrepareDeepSeekV4Inputs`, and `ModelRegistry::EncodeMm`. It runs
on a `DeepseekV4LoadedModel` that carries only the vision runtime, because
`encode_mm` reads nothing else. The oracle side is
`tools/parity/dsv4v_w6_oracle_dump.cpp`, compiled inside the pinned clone
against its own static `mtmd`. It runs llama.cpp's own `clip_init`,
`mtmd_image_preprocessor_deepseek4v::preprocess` and `clip_image_encode`, and it
sets `lead_pad` the way `mtmd.cpp:1461-1470` does. **The driver IS the oracle's
production path:** `llama-mtmd-cli`'s own `MTMD_DEBUG_EMBEDDINGS` dump, whose
tokenizer placed the image at `lead_pad = 2`, is byte-identical (`cmp`) to the
driver's `lead_pad = 2` dump. Both CPU runs are deterministic: a second job
reproduced the oracle block byte for byte.

**RESULT, for every `lead_pad` 0-3 and for the CLI's own dump** (rc job
`b69b2fb9-23b9-42b8-b755-62b8ee93b6ea`; `report-lp{0,1,2,3}.json`,
`report-cli.json`):

| Check | Result |
|---|---|
| token count, ours = oracle | 114, 115, 116, 117 for `lead_pad` 0, 1, 2, 3; 116 for the CLI |
| START, END, every NEWLINE, every PAD (leading and trailing) | **EXACT**, byte-for-byte in f32, on every rung |
| row placement (N-layout interleave) | the identity is the best cosine match for 100 of 100 image rows, on every rung |
| input pixels | ours is exactly `bf16(oracle)`; relative L2 0.12% mean |
| image rows, cosine | mean 0.99899, min 0.96709 |
| image rows, relative L2 | mean 3.83%, max 28.6% |
| image rows, absolute | mean 0.00169, max 0.0334, against a row RMS of 0.0717 |
| vit (after the final RMSNorm), 784 rows | mean cosine 0.99934, min 0.94498, mean relative L2 2.45% |

The sentinels are exact because both sides copy the same f32 vectors, and our
bf16 narrowing at the join is exact on them: the file stores them as widened
bf16. The worst "28.6%" row is a LOW-NORM row: its reference RMS is 0.0252, a
third of the block's, and its absolute error (mean 0.0057) is ordinary.

**LOCALISATION**, rc job `2481ad2a-c109-4002-8ee6-13634a2bd7f5`
(`tools/parity/dsv4v_w6_floor.sh`; `report-floor.json`, `report-samein.json`):

| Comparison, 100 image rows | cells mean rel L2 | cells mean cos | cells min cos | vit mean rel L2 |
|---|---|---|---|---|
| FLOOR: oracle(f32 input) vs oracle(bf16-rounded input) | 1.57% | 0.99986 | 0.99795 | 1.00% |
| SAME INPUT: ours vs oracle(bf16-rounded input) | 3.06% | 0.99924 | 0.97306 | 1.90% |
| AS SHIPPED: ours vs oracle(f32 input) | 3.83% | 0.99899 | 0.96709 | 2.45% |

1. **The tower amplifies a small perturbation about fifteenfold.** Rounding the
   input by about 0.1% (the bf16 step) moves the oracle's OWN output by 1.6%. A
   few percent is what a precision difference of this size produces in this
   network, so the raw number is not by itself evidence of a defect.
2. **With the input rounding taken out, ours is about twice the floor.** The
   remaining difference is inside the tower.
3. **The error has no positional structure.** Absolute error does not correlate
   with row norm (r = -0.006 as shipped, -0.06 at the same input, -0.015 on the
   floor). The spread across the ten aligner rows and the ten aligner columns is
   about 3x on the floor itself, and no row or column stands out beyond that. A
   RoPE-axis, unfold-order or padded-edge defect would load one axis.
4. **The error does not jump at a stage.** It is present at the ViT output
   (1.90%) and grows smoothly through the aligner (3.06%).

**THE TWO SIDES DO NOT COMPUTE IN THE SAME PRECISION**, and that is the
remaining candidate. At the pin, llama.cpp's CPU clip graph rounds to bf16 only
at each GEMM input (`ggml/src/ggml-cpu/ggml-cpu.c:395-399`,
`vec_dot_type = GGML_TYPE_BF16`) and keeps its residual stream, norms, RoPE,
activations and patch merge in f32. Its attention has two branches
(`tools/mtmd/clip.cpp::clip_graph::build_attn`, `:773-808`): the flash branch
casts K and V to F16 and accumulates at `GGML_PREC_F32`, and the other branch
computes QK^T, the softmax and the weighted sum in f32. `AUTO` becomes
`ENABLED` only inside `warmup()` (`:3699-3701`), and the log line
`flash attention is enabled|disabled` (`:3732`) says which branch ran. Every
oracle run here logged `warmup: flash attention is enabled`, so the oracle's K
and V were F16. Our tower carries every intermediate in bf16, which is the model
dtype `AGENTS.md` §"Inherit vLLM defaults" requires and which the model author's
own torch runtime uses.

**THE DTYPE TEST SETTLES IT.** rc job `0edbd4a9-cfd4-47e6-806b-0eb31df06383`
(`tools/parity/dsv4v_w6_f32.sh`; `report-f32.json`, `report-selfdt.json`). W2
refuses `compute_dtype != bf16`, so the job deleted that guard IN AN EXTRACTED
SCRATCH COPY ONLY. It ran the probe's f32 arm (`DSV4V_PROBE_F32=1`): the same
tower, with weights widened exactly from the file's bf16, f32 activations and the
exact f32 pixels. Nothing in `src/` changed. A bf16 control from the SAME patched
binary reproduced the first run's block byte for byte (`cmp`), so the scratch
patch changed nothing on the production path.

| Comparison, 100 image rows | cells mean rel L2 | cells mean cos | cells min cos | worst row rel L2 | vit mean rel L2 |
|---|---|---|---|---|---|
| OURS IN F32 vs oracle(f32 input) | **1.34%** | 0.99986 | 0.99666 | 8.4% | 0.88% |
| oracle's own floor (above) | 1.57% | 0.99986 | 0.99795 | 8.2% | 1.00% |
| ours bf16 vs OURS IN F32 | 3.34% | 0.99928 | 0.98445 | 20.0% | 2.07% |
| ours bf16 vs oracle (as shipped) | 3.83% | 0.99899 | 0.96709 | 28.6% | 2.45% |

**VERDICT: PRECISION, NOT A DEFECT.** Computed in f32, our tower is closer to
llama.cpp (1.34%) than llama.cpp is to ITSELF when its input moves by one bf16
step (1.57%), at both the ViT stage and the cells. The residual is llama.cpp's
own precision: bf16 GEMM inputs and F16 K/V. The shipped gap is our bf16
intermediate storage, and all of it: our bf16 output is as far from our own f32
output (3.34%) as it is from llama.cpp (3.83%). The layout, the permutation, the
2-D RoPE, the unfold order, the aligner and the four sentinels are all right on
the real weights. No defect was found, and none of the numbers above is left
unexplained.

**THE BOUND, and where it comes from.** The gate for this tower against
`llama-cpp-dsv4vision` is three conditions, and each one comes from a measurement:

1. The four sentinel kinds are EXACT and every image row is in its place. They
   are copies and a permutation, so any error at all is a defect.
2. THE F32 ARM is within the oracle's own floor: cells mean relative L2
   `<= 1.57%`, measured 1.34%. This is the condition that tests the function.
3. THE SHIPPED bf16 path is no farther from the oracle than it is from its own
   f32 arm plus that floor: cells mean relative L2 `<= 3.34% + 1.57% = 4.9%` and
   mean cosine `>= 0.998`, measured 3.83% and 0.99899. This says the bf16 path
   is the f32 function rounded, and nothing else.

A single per-row threshold is NOT the bound. The worst row's relative error
tracks its NORM, not its position. Relative error correlates with row norm at
r = -0.42 to -0.52 in every comparison, while absolute error does not
correlate with it at all (r = -0.006 to -0.12). The worst five rows come from
a small recurring set of low-norm cells: 26, 36, 69 and 82 are among the worst
five of the oracle's OWN floor, and cell 6 (row RMS 0.025, a third of the
block's) heads every other comparison's list. A relative bound on those rows
would measure the norm, not the tower.

**WHAT THIS DOES NOT SHOW.** It is one image at one size on the CPU provider. It
does not gate the device paths (W7). It does not gate what the language model
does with the block, because no DeepSeek-V4 request can be served until #2455
lands. The model author's own runtime was not run. llama.cpp is the secondary
oracle, and the f32 arm is what makes the comparison decisive without it.

### W5 evidence — the request path, and what each mutation proved

W5 makes a USER able to send an image. Five production call sites carry it, and
each is proved by deleting or inverting it. Every mutation ran in a scratch copy
and the tree was restored byte-for-byte and verified with `sha256sum -c`.

THE CHAIN, from the entry point down:

1. `LoadedEngine::FromModelDir` loads the language `.gguf` and the `--mmproj`
   second file (the W4 chain);
2. `InstallMultiModalChatSeam` -- the ONE production caller of
   `set_multimodal_chat_fn`, reached from `server_main.cpp` and, since W5, from
   `EnsureChatServing` on the `vllm_chat` path -- resolves the architecture in
   `MultiModalChatRegistry` and builds `MakeDeepSeekV4ChatSeam`;
3. `OpenAIServingChat::create_chat_completion` calls the installed seam, which
   runs `EncodeDeepSeekV4Messages`, `DeepSeekV4ImageProcessor::ProcessImage`,
   `HashImage` and `PrepareDeepSeekV4Inputs`;
4. the engine's `generate(MultiModalInputs, ...)` overload carries the features
   onto the request, and the runner reaches `ModelRegistry::EncodeMm`,
   `EmbedMm` and `Forward`.

| Mutation | Result |
|---|---|
| `feature.mm_hash = item.content_hash` (Qwen3-VL's content-only key) | RED, 2 assertions: one image at two offsets got ONE key |
| the three chunk-atomicity predicates back to their pre-W5 silence | RED, the forward THREW NOTHING and the two unit cases did not throw |
| the trailing `pad_run` rule deleted (W4/W5 reconciliation) | RED, 2 assertions: a step ending on the leading pads of the NEXT block was refused by NOTHING and returned an empty span list |
| `gather_block_table` handed the -1 no-full-attention-group sentinel again | RED, 2 assertions: the served image request died in the engine loop with `std::length_error` instead of reaching the forward |
| the codec catch re-throwing `std::runtime_error` | RED, 3 assertions: the refusals reached the client as 500 |
| `REGISTER_VLLM_MM_CHAT` repointed at an architecture nothing loads | RED, 14 assertions across every case in the suite |
| the pinned C-ABI contract case, before its own rewrite | RED, `REQUIRE( st == VLLM_OK )` -- the ABI now refuses by name |
| the `InstallMultiModalChatSeam` call deleted from `EnsureChatServing` | RED, 4 assertions: the C ABI dropped the image again |
| `mm_ctx.mmproj_path = args.mmproj_path` deleted from `server_main.cpp` | RED, 4 assertions through the real `VllmServerMain` |
| `mm_ctx.config = &loaded->config()` deleted from `server_main.cpp` | RED, 5 assertions across both serve cases |

THE LAST TWO ARE THE ONES THIS WAVE NEARLY SHIPPED UNHELD. The two fields added
to `MultiModalChatContext` are assigned in `server_main.cpp` and nowhere else
on the server path, and `test_deepseek_v4_mm_chat` fills the context in itself,
so deleting either left it green. That is the UNPASSED PARAMETER shape
`.agents/reachability.md` names.
`tests/vllm/entrypoints/openai/test_serve_deepseek_v4_mm.cpp` holds them
through the real `VllmServerMain`, in the subprocess harness
`test_serve_kv_cache_dtype.cpp` uses.

The RED-BEFORE for the seam itself was a compile failure naming the three
surfaces the wave adds: `oai::DefaultImageCodec`,
`MultiModalChatContext::config` and `MultiModalChatContext::mmproj_path`.

WHAT THE ORDER CASE MEASURES, because a count would not. Two images with
DIFFERENT grids (10x10 aligner cells against 14x14), DIFFERENT content, and TEXT
between them. A swap changes both span lengths, both keys, and the five tokens
between the spans; a fixture with two identical images could express none of the
three, and `build_image_block`'s `compress_pad = 3 - offset % 4` means the two
lengths differ at the two offsets even for one image.

THREE DEFECTS FOUND WHILE GATING, each invisible in production:

1. **Every feature carried an empty `mm_hash`.** The scheduler's per-step dedup,
   the `EncoderCacheManager` and the runner's `encoder_cache_` are all keyed on
   that string alone, so every DeepSeek image in the process was the same image:
   a second placeholder never ran the tower and was filled with the first
   image's rows. Nothing raises. The key is now the shared hasher's digest over
   the raw bytes PLUS the grid and the block's leading compression padding,
   because the encoder output is a function of `(content, grid, offset mod 4)`
   and a content-only key would splice a block of the wrong length.
2. **An interior prefill chunk returned zero spans.** W4 refused a chunk with a
   start and no end, and one with an end and no start; a chunk cut from the
   MIDDLE of a block carries neither. The visible-row rule then fell back to the
   ordinary sliding window over image rows AND the paged arm's refusal, keyed on
   a non-empty span list, did not fire, while the routing bias still applied
   because it reads the identifiers. The rule is now accounting over every media
   row, and it had to allow the leading `compress_pad` rows, which sit BEFORE
   the start identifier -- the first version refused every correct prompt and
   the existing W4 cases caught it.
3. **A PNG or `http(s)` image reached the client as HTTP 500.** The codec and
   the data-URI decoder throw `std::runtime_error`, which `api_server.cpp:373`
   maps to "InternalServerError". The DeepSeek seam re-throws them as
   `InputValidationError`. The other two seams still answer 500, which `## Owed`
   records.

TWO SEAM FIELDS WERE ADDED, because `MultiModalChatContext` could not represent
a two-file GGUF vehicle. `config` is the engine's RESOLVED model config, since
`config_path` names no file for a `.gguf` and this processor is keyed on
`vocab_size` -- it spells every image position `vocab_size + type`, so a guessed
default would put the sentinels inside the vocabulary. `mmproj_path` is the
second file, and it is the only thing at install time that can say whether the
vision half arrived, because `DeepseekV4ForCausalLM` names both the text
checkpoint and the Flash-Vision one. Without it a tower-free load refuses inside
`encode_mm`, which runs in the engine's busy loop: that stops `AsyncLLM` and
500s every LATER request, text ones included.

### W5 gate totals

On a Release CPU build with `-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SERVER=ON`:

| Suite | Cases | Assertions | Was |
|---|---|---|---|
| `test_deepseek_v4_mm_chat` (new) | 8 | 650 | -- |
| `test_serve_deepseek_v4_mm` (new) | 2 | 21 | -- |
| `test_deepseek_v4_image_processor` | 23 | 128 | 20 / 112 |
| `test_deepseek_v4_dsa` | 19 | 109 | 19 / 106 |
| `test_deepseek_v4_mm_reach` | 14 | 110 | 13 / 79 |
| `test_capi` | 69 | 685 | 69 / 676 |

`ctest -R 'deepseek_v4|clip_mmproj_gguf' -E cuda` is 26 of 26, two more suites
than W4's 24. `ctest -R 'capi|chat_mm|api_server|serving|model_registry|
model_loader' -E cuda` is 11 of 11, which is where the Qwen3-VL and dots3-note
seams are held byte-unchanged: `test_chat_mm` 11/126, `test_openai_api_server_
mm_forward` 9/73, `test_openai_api_server_dots3_mm_forward` 28/16467 and
`test_openai_serving` 48/1365 all keep their exact counts.

### W4 evidence — stage 4, the vision routing bias

The DECISION and its argument are recorded above, beside the `exp_probs_b_vl`
entry in `## Owed`, because that is where the premise this wave was handed was
corrected. This section records what was measured.

Two SELECTION cases in `test_deepseek_v4_moe` are built so the text bias, the
vision bias and the hash table each name a DIFFERENT expert, so a bias that
changed the weights but not the choice would be invisible: a mixed row of three
tokens routes text, image, text onto the text bias, the vision bias and the text
bias again, and on a hash layer a text row keeps `tid2eid` while an image row
leaves it. An EMPTY mask is asserted byte-identical to the call that carries no
vision bias at all, which is every text step.

At the forward, two language files differ in NOTHING but the VALUES of
`exp_probs_b_vl`, and the same image runs through both. The row read for logits
is a TEXT row BEFORE the image span, whose causal prefix is text only: under the
per-token rule it cannot move, and under llama.cpp's per-ubatch rule it would.
That assertion is what makes the choice a decision rather than a preference.

| Mutation | Result |
|---|---|
| llama.cpp's per-UBATCH selection (`media = any_media`) | RED: the pre-span text row moves by 16 logits, and 5 assertions in the selection cases |
| the vision bias is never selected | RED: the post-span row stops moving |
| the bias is keyed on the CHECKPOINT rather than on the row | RED: 16 logits on both the per-token case and the inertness case |

### W4 evidence — stage 2, the projector refusal order

`RefuseDeepSeekV4ClipMmprojArm` holds the order in one function and
`model_loader.cpp` calls that function rather than its parts, so a second call
site cannot get it wrong. The gate drives `LoadedEngine::FromModelDir` -- where
a user meets the message -- with a projector carrying the FUSED
`v.blk.{bid}.attn_qkv` the pinned `convert_hf_to_gguf.py` actually emits, and
asserts the message names `attn_qkv` and issue #2411 rather than blaming the
file for carrying tensors the reader never reads. Reversing the two calls
reddens it on two assertions.

### W4 evidence — stage 5, text inertness

A DeepSeek-V4 TEXT checkpoint must be exactly what it was before this wave, and
that cannot be checked against code that no longer exists. It is checked against
the OTHER checkpoint instead: a vision file differs from a text file in nothing
but its 43 `exp_probs_b_vl` tensors and its projector, and a text-only prompt
gets the same logits from both, bit for bit. The case also asserts that the text
model's `has_vision()` is false and that its `gate_bias_vl` is empty, so the two
files really are different files.

THE MUTATION IT ANSWERS TO is a real shape of the defect rather than an
arbitrary edit: key the vision bias on the CHECKPOINT -- `!L.gate_bias_vl.empty()`
-- instead of on the row's identifier. The bias is a property of the file and
the rows are not, and the two are easy to confuse. It reddens the inertness case
by 16 logits and the per-token case beside it by the same 16.

### W4 evidence — stage 3, image-span attention visibility

`deepseek4.attention.sliding_window` is 128 and one image block reaches 384
tokens, so a window applied inside a span hides more than half of it. The rule
is stated twice upstream and W4 ports both statements as ONE index rule:
llama.cpp's `swa_full_non_causal` skips the window mask at and above the span
start and applies it normally below, and the model author writes the same thing
as an index list in `get_window_topk_idxs_visible` (`inference/model.py:289-299`).

`DeepseekV4ImageSpans` reads the spans from the step's OWN identifiers -- the
processor writes `vocab_size + kImageStart` and `... + kImageEnd` -- so no new
forward channel is needed and a text step derives none. A span not closed inside
the step is REFUSED rather than truncated, because a half-visible span answers
fluently.

`DeepseekV4VisibleRows` is the whole rule and it is gated on its INDICES, at the
released numbers: a 384-token span, a 128-token window, 2000 keys. A causal-only
implementation gives a query ten rows into the span ELEVEN visible span rows
instead of 384, and the case states that number so the assertion is a
measurement rather than a restatement. Three more index cases hold the other
edges: the window still clips below the span start, the exemption does NOT leak
to a query after the span, and with no span and no window the list is the dense
causal one this branch always built.

THE FORWARD READS IT, and two cases say so through `ModelRegistry::Forward`. A
row EARLY in the span is asked for logits while a LATE row of the same span is
perturbed -- causally invisible, so only the span rule can carry it -- with a
CONTROL that perturbs a row outside the span and must not move it. And two
models differing in nothing but `attention.sliding_window` answer the same
prompt: with a window of four a token eleven rows back cannot be seen, and with
the key absent it can.

THIS BRANCH IGNORED THE WINDOW BEFORE W4, and that is a correctness change
rather than a side effect. `#2323` already recorded the same divergence for the
paged arm -- "attending the full prefix there diverges above the window" -- and
the host arm now derives the same value the paged arm does, including the
full-prefix exception for a layer with a compressor. No existing gate
distinguished the two, which is why the new case exists.

Four mutations, each restored byte-for-byte and verified with `sha256sum -c`:

| Mutation | Result |
|---|---|
| the span rule is never applied (dense causal only) | RED, 2 index cases / 5 assertions, and the forward case |
| the window term is dropped | RED, 2 index cases / 7 assertions |
| the forward never derives the spans | RED, the forward case; the index suite stays green, correctly, because it is a pure-function suite |
| the forward passes window 0, which is the pre-W4 behaviour | RED, the window case, 16 logits |

### W4 evidence — stage 1, reachability

W4 makes an image reach `ModelRegistry::Forward`. Four production call sites
carry it, and each one is proved by deleting it.

The RED-BEFORE was a compile failure. `tests/vllm/models/test_deepseek_v4_mm_reach.cpp`
was written first and named the three surfaces this wave adds:

```text
test_deepseek_v4_mm_reach.cpp:131: error: 'struct vllm::ModelSource' has no member named 'mmproj'
test_deepseek_v4_mm_reach.cpp:132: error: 'struct vllm::ModelSource' has no member named 'mmproj_path'
test_deepseek_v4_mm_reach.cpp:327: error: 'DeepseekV4LoadedModel' is not a member of 'vllm'
```

THE CHAIN, from the entry point down:

1. `LoadedEngine::FromModelDir` branches on `clip.projector_type` and calls
   `RefuseDeepSeekV4ClipMmprojArm` before the tokenizer;
2. it sets `ModelSource::mmproj`, which `LoadDeepseekV4ForCausalLM` reads;
3. that hook calls `LoadDeepseekV4VisionRuntime`, which runs
   `LoadDeepSeekV4ClipMmprojArm` and attaches the projector to the model;
4. `ModelRegistry::EncodeMm` builds the tower and runs it,
   `ModelRegistry::EmbedMm` merges its rows, and
   `ForwardDeepseekV4ForCausalLM` consumes `MultiModalForwardInput::inputs_embeds`.

THE REACHABILITY MUTATION, and it is the headline. Disabling the `input.mm`
branch in `ForwardDeepseekV4ForCausalLM` turns the focused gate RED:

```text
test_deepseek_v4_mm_reach.cpp:372: ERROR: test case THREW exception:
  vt: token id out of range at src/vllm/model_executor/models/deepseek_v4.cpp:2958
```

That is the predicted failure and not an incidental one. The expanded prompt
spells every image position `vocab_size + type`, so a forward with no merged
embeddings cannot answer the step from the embedding table at all.

Three further call-site mutations, each restored byte-for-byte and verified with
`sha256sum -c`:

| Mutation | Result |
|---|---|
| the loader's projector-type branch never selects the DeepSeek arm | RED, 4 assertions |
| the loader sets `gguf_source.mmproj = nullptr` | RED, 3 assertions |
| `LoadDeepseekV4ForCausalLM` never builds the vision runtime | RED, 3 cases |

The third mutation was GREEN on the first attempt, and that was the finding: no
case drove a load past the tokenizer, so the one line handing the projector down
was unobserved. The fixture gained `tokenizer.ggml.*` keys and a case that pairs
a projector of the WRONG aligner width with the language model, whose refusal
exists only if the file arrived.

### W4 gate totals

One place, so no section carries a number that another edit makes stale. On a
Release CPU build with `-DVLLM_CPP_CUDA=OFF` at the end of W4:

| Suite | Cases | Assertions |
|---|---|---|
| `test_deepseek_v4_mm_reach` (new) | 19 | 142 |
| `test_deepseek_v4_dsa` | 19 | 109 |
| `test_deepseek_v4_moe` | 14 | 731 |
| `test_deepseek_v4_vision` | 15 | 7412 |
| `test_deepseek_v4_mmproj` | 20 | 2211 |
| `test_clip_mmproj_gguf` | 9 | 272 |
| `test_deepseek_v4_mm_loader` | 9 | 105 |
| `test_deepseek_v4_encoding` | 21 | 54 |
| `test_deepseek_v4_image_processor` | 20 | 112 |
| `test_deepseek_v4_scaffold` | 10 | 684 |
| `test_model_registry` | 24 | 993 |

The six suites this row already owned keep their counts exactly. `ctest -R
'deepseek_v4|clip_mmproj|model_registry|model_loader'` is 28 of 28, with
`test_cuda_deepseek_v4` skipped for want of a CUDA backend.

**THE W4 REPAIR ROUND MOVED TWO OF THOSE ROWS, and the table above already
carries the new numbers.** `test_deepseek_v4_mm_reach` went from 13 cases and
79 assertions to 19 and 142; `test_deepseek_v4_dsa` kept its 19 cases and went
from 106 assertions to 109. Every other row is unchanged and
`ctest -R 'deepseek_v4|clip_mmproj_gguf' -E cuda` is 24 of 24. What the six new
cases hold is listed in `## Owed` and in each repair commit: the projector
pairing refusal, the paged arm in the direction that must NOT refuse, the
`supports_multimodal` flip's effect on the chat seam, the two image-row
predicates agreeing, the hash-layer skip at a forward, and a chunk that opens no
image span. The aligner-row permutation assertion was also VACUOUS until this
round -- every aligner row of the fixture was bit-identical -- and its repair is
the one change here that alters an existing case rather than adding one.

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

`ACTIVE`. W1, W2, W3, W4 and W5 have landed on the row branch, W6 ran the first
real-weight gates, and W7-CUDA has run the first DEVICE ones.

W7-CUDA IS THE WAVE THAT PUT THE VISION TOWER ON A GPU. Before it, the tower
could not take a CUDA queue at all: the `deepseek4v` mmproj reader left every
weight a host view — its own comment said "W4 owns the upload" and W4 did the
routing instead — so `ValidateQueue` refused every device queue, and a SERVED
image request died with that same sentence rather than with the `fp8_ds_mla` KV
blocker everyone expected. `EnsureResident` stages the tower on first use and
rebuilds the MLP gate-up borrows against the staged tensors; the host-to-device
case is added and the device-to-device case is still refused by name, so the
refusal is narrowed rather than deleted. On `thor:gpu0` (sm_110) the tower then
ran on the real 934,462,656-byte projector at every `lead_pad` rung and matched
llama.cpp `b10766` DIRECTLY: the four sentinel kinds byte-exact, the identity
permutation best for 100 of 100 rows, cells at 2.884% mean relative L2 and
0.99939 mean cosine — inside W6's declared bound of 4.9% and 0.998, and closer
to the oracle than our own CPU arm. W4's windowed `dev_attn` refusal, which no
CPU build could execute, FIRED and is now measured. The numbers, the job ids and
what stays unmeasured are in `### W7-CUDA evidence`.

WHAT W7-CUDA DID NOT DO. No image has been ANSWERED yet: the served request now
gets past the tower and dies at `vt: MatVec weight size mismatch`, which is a
different, unexplained defect. The `dev_attn` image-span refusal and the two
device routers' media refusal were never driven, so both stay unmeasured. Eight
`mm_reach` cases fail on aarch64 for a host-side quant-repack reason proven by
an A/B, not a device one. `test_serve_deepseek_v4_mm` times out with no output
and nothing here explains it. No speed was measured. ROCm and Vulkan are
untouched. Every one of these is under `## Owed`.

W6 IS THE FIRST TIME THE REAL WEIGHTS RAN, and the vision half is right. On
`thor:gpu0`, the pinned `UD-IQ1_S` language model loads and generates text, and
the real `mmproj-BF16.gguf` runs through `ModelRegistry::EncodeMm`. Its token
block matches llama.cpp `b10766` on a 392x392 image that neither side resamples.
The token count is the same for all four leading-pad offsets, and the four
sentinel kinds are byte-exact. Every image row is in its place. The image rows
agree at a mean cosine of 0.99899. That gap is our bf16 intermediate storage
and nothing else: the same tower in f32 lands closer to llama.cpp than llama.cpp
lands to itself under a one-bf16-step input change. `llama-cpp-dsv4vision` built
and ran the model end to end, so it is now `gateable = yes`. The measurements,
the job ids and the bound are in `### W6 evidence`.

WHAT W6 DID NOT DO. It could not serve a single request: the server exits at
engine start on the `fp8_ds_mla` KV cache, which `KV-DSV4-MULTICACHE` W5 owns
(#2455). No image answer has come from this engine, and a CUDA build would
still refuse an image step in the device routers, which W7 owns. The text
generation is plausible but not oracle-gated. No speed was measured. Both are
under `## Owed`.

W5 IS THE WAVE THAT MADE A USER ABLE TO SEND AN IMAGE. W4 made one reach
`ModelRegistry::Forward`; every seam above it was still unwired, and
`MultiModalChatRegistry::Find("DeepseekV4ForCausalLM")` was null, so the
server's install answered every image request for this architecture with a
REFUSING seam. `src/vllm/entrypoints/openai/mm_chat_deepseek_v4.cpp` registers
the factory, and its chat function runs the pinned `encode_messages` port, the
W1 image processor and `PrepareDeepSeekV4Inputs` on every image request.
Several interleaved images are served in source order with the ceiling coming
from `MultiModalConfig`. `vllm_chat` installs the same seam, so the capability
is on `include/vllm.h` and the server is a client of it rather than the only
door.

Three defects were found and fixed on the way, each of which would have been
invisible in production. Every feature carried an EMPTY `mm_hash`, which the
scheduler and both encoder caches key on, so a second image in one request
never ran the tower and was filled with the first one's rows. An INTERIOR
prefill chunk of an image block carried neither structural identifier, so
`DeepseekV4ImageSpans` returned zero spans and the step was served from the
ordinary sliding window with the paged-arm refusal unarmed. And a PNG or
`http(s)` image reached the client as HTTP 500 rather than 400.

WHAT W5 DID NOT DO. `ParseDeepSeekV4TaggedText` is still unreached, image
prefill is still not atomic at the scheduler, the container codec is still
refused rather than implemented, and no served request can GENERATE on a CPU
build because the runner's gather-logits path reaches
`DeepseekV4Model::ForwardDevice`. All four are named under `## Owed` above.
When W5 landed no real artifact had been read or run. W6 has since read and run
both files; see `## Now` and `### W6 evidence`. W7 owns the device paths.

THE W4/W5 MERGE CARRIED A REDUNDANT REFUSAL, and it is removed. Both waves
closed the chunk-atomicity gap from opposite sides and the merge took both, so
`DeepseekV4ImageSpans` ran a `spans.empty()` loop refusing any out-of-vocabulary
row AHEAD of the trailing `pad_run` rule. That loop refused nothing the trailing
rule does not: an empty span list means no START identifier was ever read, so
`pad_run` was never reset and every pad the step carries is still counted at the
end -- its condition is a SUBSET of the trailing rule's. Its only effect was to
answer FIRST and with a different message, and that is what left the trailing
rule ungated: deleting the trailing rule kept every case in the suite green. The
loop is gone, and `test_deepseek_v4_dsa` now drives the pad-only chunk it was
shadowing.

SETTLED BY CONSTRUCTION AND NOT BY READING. Every token sequence of length 1 to
6 over `{two ordinary ids, kImageStart, kImagePad, kImage, kImageNewLine,
kImageEnd}` -- 137,256 of them -- was run through `DeepseekV4ImageSpans` with
the loop present and with it deleted. The two agree on every single one: 134,405
refuse either way, and the 2,851 that do not return the same span count. There
is no input that reaches the loop and nothing else.

The merge also pointed one case at a message that no longer fires --
`test_deepseek_v4_mm_reach`'s interior chunk is refused where the loop READS the
row, not after it -- and left one asserting that this architecture has no
registered chat seam, which W5 gave it. Both are replaced rather than repaired,
because each pinned a premise another wave superseded.

W4 IS THE WAVE THAT MADE THE ROW REACHABLE. An image now travels from
`--mmproj` through `ModelSource::mmproj` into `LoadDeepseekV4ForCausalLM`, which
attaches the `deepseek4v` projector to the model; `ModelRegistry::EncodeMm` runs
the W2 tower and emits one row per sentinel token, `ModelRegistry::EmbedMm`
merges those rows over the image span, and the registered forward consumes
`MultiModalForwardInput::inputs_embeds`. The two language-side behaviours the
original spec missed are implemented with it: the vision routing bias is
selected PER TOKEN, with the argument for that divergence recorded above, and an
image span attends across itself while the window still clips below its start.

WHAT W4 DID NOT DO, AND W5 DID. The REQUEST path was unwired -- nothing between
an HTTP body and `MultiModalInputs` called the W1 encoder or processor -- and
W5 wired it; see `## Now` above.

FIVE arms still refuse rather than serve a step they would answer wrongly, and
every one is listed under `## Owed`: the paged attention arms and the two device
routers refuse an image step, the device decode attention kernel refuses an image
step AND a windowed one, and `DeepseekV4ImageSpans` refuses a prefill chunk that
carries part of an image block without its markers. The device decode refusal is
the one no CPU build can execute, and its entry says so.

When W4 landed no real artifact had been read or run by this code. W6 has since
loaded both files and compared the vision block with the oracle; W7 owns the
device paths.

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
Green after at `78ecaf034`: 15 of 15 cases and 7407 of 7407 assertions with the
tree restored. `aligner_hidden` keeps its absolute cap, so the stage stays transitively bounded
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
Green after at `18e6aad8b`: 15 of 15 cases and 7409 of 7409 assertions with the
tree restored and verified by SHA-256.

What the cap does NOT see is a new buffer the pool serves from a block that was
already free, which adds no driver allocation and no retained bytes. That is
narrower than the withdrawn claim and is stated rather than assumed.

A second review found the specific buffer the list had always missed:
`DeepSeekV4Vision::Forward`'s own `vision` DBuf, the tower output it hands to
the aligner. Widening it was already caught, by the shape and dtype check
`VisionForward` runs on the tensor it is passed, so nothing was unguarded; but
the capture struct said the list held one entry per internal scratch buffer and
it did not. It is now recorded as `forward.vision`, BETWEEN the two stages
rather than first, because the vision stage is what clears the list and a record
ahead of it would be erased. Deleting that one call reds the case on
`REQUIRE(scratch.size() == expected.size())`. The struct's comment now says the
list is a declaration rather than a measurement, and points at the pool cap for
the buffers a declaration cannot cover.

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
SUPERSEDED: that sentence landed in `c211c50fd`, the same commit that added a
fifteenth case, so it was stale on arrival. The two guards are unchanged and
still non-vacuous; only the totals moved, and the totals now live in one place at
the end of this section rather than beside each measurement, because a count
written beside a measurement is stale the next time anybody adds a case.

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
than a struct field. `test_deepseek_v4_mmproj` reported 19 cases and 2218
assertions at `8f44e7bbc`; see the head counts at the end of this section.

**BOUNDING THE FACTORS WAS NOT BOUNDING THE PRODUCT, and a second independent
review EXECUTED the falsification.** SUPERSEDES the claim in
`include/vllm/model_executor/models/clip_mmproj_gguf.h` that an out-of-range
value is refused with the key that carried it "rather than surfacing as
`length_error` or `bad_alloc`". It surfaced as exactly that.

`kMaxGeometry` is `1 << 20`, and the loader reserved the fused qkv buffer at
`3 * hidden * hidden` BEFORE its first file-shaped read, so nothing about the
file bounded it. At an `embedding_length` of 65536 -- a sixteenth of what the
field bound allows -- that is 12,884,901,888 elements, and at the permitted
maximum about 6.6 TB. Reproduced here rather than taken from the report: a
1.6 MB projector declaring 65536 at `patch_size` 1, carrying only
`v.patch_embd.weight`, `v.patch_embd.bias`, `v.blk.0.ln1.weight` and
`v.blk.0.ln2.weight`, passed `RefuseUnsupportedDeepSeekV4ClipMmproj`, passed
every `RequireGeometry`, and threw `std::bad_alloc` under
`ulimit -v 6000000`. Third time in this row for this defect class, one
multiplication further from the bound each time, and the `kMaxGeometry` comment
had named the shape ("the same defect one step removed") while choosing a bound
that does not contain it.

Two changes, and the second does not depend on the first. `kMaxTensorElements`
(`1 << 28`, about seven times the shipped artifact's largest tensor at
`mm.1` = 4096 * 1024 * 9 = 37,748,736 elements) now bounds the ELEMENT COUNT of
every tensor the loader materializes -- the fused qkv weight, the patch
embedding weight, the merged gate/up weight and the aligner's two projections --
on the parsed values, before anything is reserved, and names the keys whose
product produced it. `SaturatingElements` clamps rather than multiplies, because
four factors at `kMaxGeometry` is 2^80 and would wrap int64 into a small
positive number, which is the same defect one step further removed again. And
the `reserve` is gone: it ran ahead of every file-shaped read, and growing on
`insert` keeps that site bounded by the bytes `read.Bf16` returns even if the
ceiling is ever widened.

Red before: the case reports `message := std::bad_alloc` and reds three of its
assertions. Green after: the same file is refused by name with
"clip.vision.embedding_length ... 268435457 elements or more ... 268435456 per
tensor". Deleting the qkv product bound reds it again. The rest of the file was
re-read for the same shape: `patch_weight`, `gate_up` and the aligner all size
from a `read.Bf16` that has already been matched against the file, and
`blocks.resize` is bounded by `kMaxDeepSeekV4Depth`. The two remaining unbounded
sites are `vw.blocks.resize(cfg.depth)` and `patch_proj_w.assign` in the
PRODUCTION-REACHABLE Qwen3-VL arm, which
[#2995](https://github.com/mudler/vllm.cpp/issues/2995) owns and which is not
touched here.

**One assertion in the `block_count` case measured nothing.**
`CHECK(Contains(neg, "-1"))` passed with every geometry bound deleted, because
the fallback unaccounted-tensor message prints "enumerated for depth -1". So
`750cc6626`'s "failed on all four of its message assertions" read consistent
while the case carried five, one of them vacuous. Every geometry case now
asserts the whole refusal through `RefusedByBound`, which spells
"<key> is <value>, and this reader accepts 1 to ". With the `block_count` and
`embedding_length` bounds deleted, all three of that case's assertions red
rather than four of five.

**The geometry guard does NOT run on a user-supplied `--mmproj` today, and
`750cc6626` said it does.** SUPERSEDES that sentence. No file under `src/`,
`include/`, `examples/` or `tools/` calls `DeepSeekV4ClipMmprojVisionConfig`,
`RefuseUnsupportedDeepSeekV4ClipMmproj`, `RefuseUnaccountedDeepSeekV4ClipMmproj`
or `LoadDeepSeekV4VisionFromClipMmproj`. The whole deepseek4v mmproj arm is a
staged slice whose wiring `## Owed` gives to W4, so the untrusted header reaches
the guard through the test suite and nowhere else. The guard is right and W4 is
what makes the claim true. The arm where the claim is ALREADY true is the
production-reachable Qwen3-VL `ClipMmprojVisionConfig` beside it, which reads
`block_count` from a user-supplied `--mmproj` into an unbounded `resize`, and
[#2995](https://github.com/mudler/vllm.cpp/issues/2995) owns that. It is not
repaired here: it is a different arm, outside this row, and it is recorded rather
than fixed for the same reason
[#2992](https://github.com/mudler/vllm.cpp/issues/2992) is.

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

### Repair round 2 head counts

ONE PLACE, DELIBERATELY. Every count above is dated to the commit that measured
it, because a total written beside a measurement is stale the next time anybody
adds a case -- which is what happened to the W3A sentence this round superseded,
and it went stale inside the same commit that wrote it.

Measured on a CLEAN Release CPU build (`-DVLLM_CPP_CUDA=OFF`, `-j 4`, no
warnings), every run under `ulimit -v 6000000`:

| Suite | Cases | Assertions |
|---|---|---|
| `test_deepseek_v4_vision` | 15 | 7412 |
| `test_deepseek_v4_mmproj` | 20 | 2211 |
| `test_clip_mmproj_gguf` | 9 | 272 |
| `test_deepseek_v4_mm_loader` | 9 | 105 |
| `test_deepseek_v4_encoding` | 21 | 54 |
| `test_deepseek_v4_image_processor` | 20 | 112 |

`ctest -R 'deepseek_v4_(vision|encoding|image_processor|mmproj|mm_loader)|clip_mmproj_gguf'`
passes 6 of 6.
