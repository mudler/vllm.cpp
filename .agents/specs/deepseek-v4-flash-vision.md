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
- A sidecar format that combines a text GGUF with separate vision weights. The
  shipped quantized arm is one documented model artifact, not an assembly recipe
  users have to reconstruct.
- A token-only proof for the tower or attention visibility. Stage numerics and
  memory format are load-bearing because an omitted image mechanism can leave an
  argmax unchanged.

## Released artifact and geometry

The pinned Hugging Face index reports `167,811,372,792` bytes, or 156.287 GiB,
across 48 safetensors shards. It does not fit a 119 GiB GB10. The model author's
reference conversion and launch recipe use tensor parallelism 4. A single-device
production gate therefore depends on the GGUF k-quant arm; the official
safetensors arm remains a multi-device gate.

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
- llama.cpp has no released DeepSeek-V4 Vision architecture or combined GGUF
  converter. It may become the quantized speed floor only after a pinned
  implementation exists; it is not the algorithm oracle now.

Calling the model-author runtime the `transformers` oracle would be false. The
executing vision code lives in the Hugging Face checkpoint repository under
`inference/`, not in `huggingface/transformers`.

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
| released safetensors index and `inference/convert.py` | `deepseek_v4_weights.cpp` plus a separate DeepSeek-V4 Vision GGUF loader/converter | Account every official tensor and ship one combined k-quant artifact |
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

A separate DeepSeek-V4 Vision GGUF conversion and loader TU extends the existing
`deepseek4` architecture rather than creating a sidecar. Language tensors use
the shared k-quant/i-quant loader and keep-quant compute. The approximately
0.869 GiB vision/aligner/sentinel group remains BF16 in the first artifact.

Before capability publication, `docs/USAGE.md` records the exact Hugging Face
repository and revision, artifact filename, byte size, resident size and
SHA-256. The official 48-shard arm and every refused arm are listed beside the
GGUF vehicle. A third-party quant is labelled as third-party. If no combined
quant can be produced, the row stays incomplete; loader scaffolding is not
model support.

## Dependencies

- Landed `MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm`: language weights,
  DSA, MHC, MoE, KV-cache and ordinary decode.
- Landed multimodal request and device-embedding seams:
  `MultiModalInputs`, `MultiModalFeatureSpec` and `MultiModalForwardInput`.
- The registered `deepseek-v4-vision` oracle pin. Its first real run stays owed
  by #2411 and does not become a static-source pass.
- An eligible leased TP4 topology for the official arm and one device with enough
  memory for the combined GGUF arm.
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
| W3 | Official weights and combined GGUF arm | Pinned index has zero unaccounted tensors; official arm loads on its eligible topology; the documented GGUF artifact loads without a sidecar and fits one gate device |
| W4 | Merge, visibility and cached language forward | Registered forward consumes image embeddings, image-span attention matches the oracle, image prefill is atomic, decode does not rerun vision, text-only DeepSeek remains byte-identical |
| W5 | Runner, public ABI and OpenAI serving | Multiple data-URI and HTTP(S) PNG/JPEG images reach `ModelRegistry::Forward` in order; Qwen and Gemma multimodal smoke cases remain unchanged |
| W6 | Real-checkpoint correctness, speed and publication | Greedy gate passes on the pinned reference and quantized arm; TTFT, vision encode, prefill, decode and memory are recorded; user documents name exact weights |

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
  storage formats, GGUF name map and named refusals.
- `test_deepseek_v4_mm_e2e`: production `ModelRegistry::Forward` on the pinned
  image prompts, then real-checkpoint generated ids.
- `test_deepseek_v4_mm_server`: OpenAI multi-image request through the actual
  server surface, including PNG/JPEG data URIs and HTTP(S) media.

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
6. **The official artifact cannot fit one GB10.** A combined GGUF is a completion
   dependency, not an optional optimization.
7. **Remote image fetching is security-sensitive.** Reuse the shared HTTP/TLS
   transport and its timeouts; do not shell out or add a DeepSeek-only fetcher.
8. **DSpark weights are present.** Account for them and keep the optional
   speculator's lifecycle separate. Silent activation or silent dropping is
   forbidden.
9. **The upstream repository is experimental.** Every source and artifact link
   uses the 40-hex pin. A force-push or replacement checkpoint triggers the stop
   condition below.

## Stop conditions

- The pinned Hugging Face revision no longer resolves or its index/artifact
  identity changes: stop, record the new identity, and obtain a new pin decision.
- A complete oracle cannot build or run on an eligible leased topology: keep
  `gateable = no`, record the exact dependency or hardware blocker, and do not
  claim end-to-end support.
- The combined quantized arm cannot preserve the released tensor set or fit an
  available gate device: the row remains incomplete; do not publish a sidecar
  workaround as support.
- Current vLLM lands a complete implementation before W1: stop and rebase the
  design onto that exact vLLM revision rather than maintaining the model-author
  runtime as the mirror source.
- A wave requires bypassing `ModelRegistry::Forward`, the shared multimodal input
  types or the DeepSeek attention/KV seams: return `NEEDS_DECISION` with the
  unrepresentable behavior and the smallest seam extension.

## Owed

- The first TP4 oracle run and committed evidence are owed by issue #2411 and W1.
- The combined GGUF artifact, its revision and SHA-256 are owed by issue #2411
  and W3.
- W1 prompt encoding and image preprocessing remain unreachable from a
  production entry point. W4 wires them into the registered model forward, and
  W5 wires the runner, public ABI and OpenAI server for row
  `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`; issue #2411 tracks both
  waves.
- DeepSeek-V4 DSpark remains owned by
  `MODEL-SPEC-deepseek-v4-dspark-deepseek-v4-for-causal-lm`; this row only
  accounts for and names its tensors.

## Now

`ACTIVE`. W1 ports the pinned prompt encoder and image processor into the shared
`MultiModalInputs`, `ImageKwargs` and `MultiModalFeatureSpec` surface. The model
forward remains unwired: ViT, aligner, weights, image-span visibility, ABI and
server work stay in W2-W5.

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
