# SPEC — MODEL-XOR: xor SystemOne-class decision model (Qwen3.6-35B-A3B MoE + decision head)

Port `juspay/xor` into vllm.cpp as a SystemOne-class decision model,
reusing the `/v1/systemone` API (choice/score/noul question types) and
`vllm_decide` ABI (v29) already on `main` from the Laya/kev work. xor is a
35B MoE model post-trained from Qwen3.6-35B-A3B (35B total, ~3B activated
per token). It is multimodal (up to 8 images) with fully merged BF16 weights.
The Qwen3.6-35B-A3B MoE backbone does not exist in vllm.cpp for the decision
pipeline — the Qwen3.5 MoE text-generation forward exists (registered as
`Qwen3_5MoeForConditionalGeneration` in `qwen3_5_moe.cpp`), but ForwardHidden
(hidden-state extraction) for the MoE variant, multimodal vision integration
for the MoE backbone, and the decision head are all new. CPU + GPU (CUDA),
OpenAI-compatible serving through `/v1/systemone`.

## Now

`SPEC`

## Scope

- **Row.** `MODEL-XOR` (this spec). New model-matrix row under the
  SystemOne-class decision model family — xor answers structured questions
  through `/v1/systemone`, same class as GLiNER2.5, Laya, and kev.
- **In.** Qwen3.6-35B-A3B MoE backbone in feature-extraction mode
  (ForwardMoeHidden, no LM head) — this is NEW: the existing Qwen3.5/3.6 MoE
  text-generation forward exists (`qwen3_5_moe.cpp`), but hidden-state
  extraction for the MoE variant does not (only `ForwardDenseHidden` exists for
  the dense variant); multimodal image processing (up to 8 images, vision tower
  + embedding merge); deterministic single-token candidate readout (decision
  head that reads one token logit per candidate option); forward+reverse
  option-order evaluation (run the decision forward twice — options in original
  order and in reversed order — then calibrate); probability calibration
  (combine forward+reverse probabilities into final distribution); typed
  decision (noul/choice/score question types); fully merged BF16 weights (no
  LoRA, no adapter — weights are pre-merged at training time); model
  registration via `REGISTER_VLLM_MODEL`; `/v1/systemone` dispatch for xor;
  `vllm_decide` ABI (v29) integration; CPU build + tests; GPU forward (CUDA).
- **Out (owned by other rows).** ROCm kernel tuning. GGUF k-quants (owed,
  named below). Runtime prompt-activated adapters (xor ships fully merged
  weights). Prefix caching optimization (deferred — xor reprocesses state per
  question, and forward+reverse doubles that). LocalAI backend is a separate PR
  in a separate repo.
- **Reuse.** The `/v1/systemone` API endpoints, request/response structs, and
  server dispatch from GLiNER2.5/Laya/kev. The `DecisionFn` callback mechanism
  from Laya. The `vllm_decide` / `vllm_decide_free` C ABI (v29, already in
  `include/vllm.h:1192-1197`). The Qwen3.5/3.6 MoE weight loading infrastructure
  (`qwen3_5_weights.cpp`, written against `Qwen/Qwen3.6-35B-A3B` and
  `nvidia/Qwen3.6-35B-A3B-NVFP4`). The Qwen3.5/3.6 MoE forward machinery
  (`Qwen3_5Model::Forward` in `qwen3_5.cpp`, including `MoeBlock` /
  `RunMoeBlock` from `qwen3_5_moe_block.h`). The `return_hidden` branch in
  `DenseForwardLayers` (`qwen3_5.cpp:9835,9993-9995`) — the precedent for
  hidden-state extraction. The Qwen3-VL vision tower pattern (`qwen3_vl_vision.cpp`,
  `qwen3_vl_registry.cpp`) for image processing. The Qwen2 BPE tokenizer
  (already supported).

## Upstream chain

### Oracle: SGLang (validated serving runtime)

SGLang is the secondary oracle per AGENTS.md §"When vLLM has no implementation"
(§259). vLLM does not implement the xor decision model. SGLang serves xor as a
validated runtime.

Oracle pin: `sglang` @ `f63458b5beaceabbd9d749b9fc956370e1b649e6` (v0.5.15),
`gateable = yes`. See `.agents/oracles/sglang.md`.

SGLang provides:
- The serving runtime for xor (model loading, inference, API serving).
- Reference outputs for correctness cross-check (choice/score/noul, with and
  without images).
- The performance floor for equivalent workloads.

The greedy token-ID correctness cross-check (`SGLANG-ORACLE-CORRECT`) is
`INVENTORIED` at the pinned revision. If it has not been run for xor
specifically, the E2E parity gate uses distributional tolerance, not
token-exact matching (see Risks).

### Base model: Qwen3.6-35B-A3B

- 35B total parameters, ~3B activated per token (sparse MoE with top-k expert
  routing + shared expert).
- Hybrid architecture: full-attention layers + GDN (Gate DeltaNet)
  linear-attention layers (same hybrid layout as Qwen3.5 — the `qwen3_5_*`
  files handle both Qwen3.5 and Qwen3.6, confirmed by `qwen3_5_common.h:1,22`
  and `qwen3_5_moe.cpp:1`).
- MoE MLP: `MoeBlock` with routed experts (`ExpertMlp`) + shared expert
  (`SharedExpert`), exposed via `RunMoeBlock` (`qwen3_5_moe_block.h:52`).
  `moe_intermediate_size` per expert, `num_experts` routed experts.
- BF16 weights. The text-generation forward path exists in vllm.cpp as
  `Qwen3_5MoeForConditionalGeneration` (`qwen3_5_moe.cpp:256`), but
  ForwardHidden (feature-extraction / hidden-state extraction) for the MoE
  variant does NOT exist — only `ForwardDenseHidden` exists for the dense
  variant (`qwen3_5.cpp:9424`).
- `kQwen3_5Info` already declares `supports_multimodal = true`
  (`qwen3_5_common.h:29`), but no vision tower is wired for the MoE variant.

### Post-trained model: juspay/xor

- 35B MoE post-trained from Qwen3.6-35B-A3B for SystemOne-class
  decision-making.
- Decision head: deterministic single-token candidate readout — for each
  candidate option, the model reads one token position whose logit/probability
  is the candidate's score. This differs from kev's PointerHead (dot-product
  attention readout) — xor uses direct token-position logit readout.
- Forward+reverse option-order evaluation: options are evaluated in original
  order AND in reversed order, and the two passes are combined to reduce
  position bias.
- Probability calibration: forward and reverse probabilities are calibrated to
  produce the final option probabilities (exact combination method determined
  from the xor reference / SGLang oracle).
- Multimodal: up to 8 images per request, processed through a vision tower
  and merged with text embeddings.
- Fully merged BF16 weights — no adapter, no LoRA at load time.
- Question types: noul (binary yes/no), choice (select best option from K),
  score (rate on L levels).

## Design

### Phase 1: Qwen3.6 MoE backbone — ForwardMoeHidden (delta from Qwen3.5 MoE)

The Qwen3.6 MoE text-generation forward exists (`Qwen3_5Model::Forward` in
`qwen3_5.cpp`, registered in `qwen3_5_moe.cpp`). What does NOT exist is the
hidden-state extraction path for the MoE variant.

- Add `ForwardMoeHidden` to the Qwen3.6 MoE model, mirroring
  `ForwardDenseHidden` (`qwen3_5.cpp:9424`) and `Qwen3DenseModel::ForwardHidden`
  (`qwen3.cpp:570-592`).
- The forward calls the existing `Qwen3_5Model::Forward` with
  `return_hidden=true` — the `return_hidden` branch already exists in
  `DenseForwardLayers` (`qwen3_5.cpp:9835,9993-9995`), which skips `lm_head`
  and returns post-final-RMSNorm hidden states as f32 rows. The MoE forward
  path (`MoeBlock`, `RunMoeBlock`) is UNCHANGED — it already produces hidden
  states; the delta is the `return_hidden` extraction branch.
- Returns `[n_out, hidden_size]` f32 rows.
- Delta from Qwen3.5 MoE: architecturally identical (same hybrid
  full-attention + GDN layout, same MoE block structure, same weight naming).
  The delta is the ForwardHidden extraction path: `ForwardDenseHidden` exists
  for the dense variant but no equivalent exists for the MoE variant. The
  `return_hidden` flag in `DenseForwardLayers` is the precedent — it must be
  threaded through the MoE forward path identically.
- Precedent: `LlamaEmbeddingLoadedModel` uses `ForwardHidden` for pooling
  (`llama_embedding_registry.cpp:115`); kev uses `ForwardDenseHidden` for
  PointerHead readout (`kev_registry.cpp:175`).

### Phase 2: Multimodal image processing (up to 8 images)

xor accepts up to 8 images per request. The Qwen3.6 MoE backbone's
`kQwen3_5Info` already declares `supports_multimodal = true`
(`qwen3_5_common.h:29`), but no vision tower is wired for the MoE variant.

- Reuse the Qwen3-VL vision tower pattern: `Qwen3VLVisionForward`,
  `Qwen3VLVisionConfig`, `Qwen3VLVisionWeights` from `qwen3_vl_vision.cpp` /
  `qwen3_vl_vision.h`.
- Image processing pipeline: pixel values (bf16) → vision tower forward →
  feature embeddings → merge with text token embeddings via
  `_merge_multimodal_embeddings` (masked scatter, mirroring
  `qwen3_vl_registry.cpp:386`).
- Up to 8 images: the vision tower runs per-image, features are concatenated
  and merged into the text sequence at image placeholder positions. The
  `MultiModalFeatureSpec` (`vllm/multimodal/inputs.h`) carries per-image
  `pixel_values_bf16` and `image_grid_thw`.
- MRoPE (multidimensional rotary position embeddings) for image tokens,
  mirroring `Qwen3VLGetRopeIndex` (`qwen3_vl_registry.cpp:464`).
- DeepStack visual features (if the xor checkpoint carries them), mirroring
  `Qwen3VLComputeDeepstack` (`qwen3_vl_registry.cpp:388`).
- The vision tower weights are part of the xor checkpoint (fully merged BF16).
- The vision tower config is read from the xor checkpoint's `config.json`
  (nested `vision_config`), mirroring how `qwen3_vl_registry.cpp` reads
  `weights.vision_cfg`.

### Phase 3: Decision head + deterministic single-token candidate readout

xor uses a deterministic single-token candidate readout: for each candidate
option, the model reads one token position and the logit at that position is
the candidate's score.

- The decision head takes the hidden states from ForwardMoeHidden (Phase 1)
  and applies a lightweight projection (or direct logit readout) at each
  candidate's token position.
- For choice questions: each option is a candidate; the model reads the logit
  at the option's candidate token; softmax over candidates gives the option
  distribution.
- For score questions: each level is a candidate; the model reads logits at
  level tokens; softmax over levels gives the level distribution.
- For noul questions: the binary candidate (true/false) is read from a single
  token position.
- The readout is DETERMINISTIC: no sampling, no temperature — the raw logit at
  the candidate position is the score.
- This differs from kev's PointerHead (q/k projection + scaled dot-product
  attention readout) — xor uses direct token-position logit readout from the
  LM head (or a decision-specific head) at the candidate position.
- The exact readout mechanism (which token position, which vocabulary token,
  whether the LM head or a separate decision head is used) is determined from
  the xor reference / SGLang oracle.

### Phase 4: Forward+reverse option-order evaluation + probability calibration

- For each question, run the decision forward TWICE:
  1. Forward pass: options in original order [o1, o2, ..., oK].
  2. Reverse pass: options in reversed order [oK, ..., o2, o1].
- Each pass produces a probability distribution over options (from Phase 3's
  candidate readout).
- Probability calibration: combine forward and reverse probabilities to
  produce the final calibrated distribution. The combination method
  (geometric mean, arithmetic mean, or xor-specific formula) is determined
  from the xor reference / SGLang oracle.
- The calibrated probabilities produce:
  - choice: argmax option + confidence
  - score: expected score + confidence
  - noul: P(true) + confidence
- Confidence formulas (to be confirmed against SGLang oracle, but expected to
  match the shared SystemOne formulas already on `main`):
  - choice: `(max(p) - 1/K) / (1 - 1/K)` (already on main as
    `ChoiceConfidence`)
  - score: `1 - E|level - mode| / (L - 1)` (already on main as
    `ScoreConfidence`)
  - noul: `p(true)`, no confidence
- All probabilities rounded to 2 decimals (same as kev/Laya).
- Forward+reverse doubles inference cost per question. For a 35B MoE, this is
  significant but inherent to the xor method. Prefix caching of the shared
  state prefix across the two passes is deferred.

### Phase 5: Registration + /v1/systemone dispatch

- Register via `REGISTER_VLLM_MODEL` (mirror `kev_registry.cpp`,
  `laya_registry.cpp`).
- `LoadedModel` subclass owning merged Qwen3.6 MoE weights + vision tower
  weights + decision head weights.
- `is_pooling_model=true`, `is_text_generation_model=false` (feature-extraction
  mode).
- `/v1/systemone` dispatch: reuse the `DecisionFn` callback mechanism from
  Laya/kev. xor plugs in as an alternative model behind the same API.
- `vllm_decide` ABI (v29, `include/vllm.h:1192-1197`): the engine's
  `vllm_decide` entry point dispatches to the xor decision pipeline when the
  loaded model is xor. The request JSON is the raw body of `POST /v1/systemone`;
  the response is the JSON result string (caller frees with `vllm_decide_free`).
- The xor model handles multimodal inputs: image data is passed through
  `ModelForwardInput.mm` (the multimodal feature spec), mirroring
  `qwen3_vl_registry.cpp`'s `EmbedMultiModal` and `ComputeMmRope` paths.

### Phase 6: GPU (CUDA)

- GPU forward for the MoE backbone: the existing `Qwen3_5Model::ForwardDevice`
  path handles MoE on CUDA (`qwen3_5_moe.cpp:210`). The `ForwardMoeHidden`
  path adds the `return_hidden` branch to the device forward, mirroring how
  `ForwardDenseHidden` relates to the dense device path.
- GPU forward for the vision tower: `Qwen3VLVisionForward` already has a device
  path with `Qwen3VLVisionDeviceWeights` (`qwen3_vl_vision.cpp:197,307`).
- The decision head forward is lightweight (logit readout) and runs on host or
  device.
- No new CUDA kernels expected — xor routes through existing `vt::` ops and
  the existing MoE/vision device paths.

## Our baseline

Before this row: the Qwen3.6 MoE text-generation forward exists
(`qwen3_5_moe.cpp`, `Qwen3_5Model::Forward`), but ForwardHidden (hidden-state
extraction) for the MoE variant does not — only `ForwardDenseHidden` exists
for the dense variant. The Qwen3-VL vision tower exists but is wired for
`Qwen3VLForConditionalGeneration` (a separate model), not the Qwen3.6 MoE.
The `/v1/systemone` API, `DecisionFn` callback, and `vllm_decide` C ABI (v29)
exist from the Laya/kev work. No decision head with deterministic single-token
candidate readout exists. No forward+reverse option-order evaluation exists.
No probability calibration exists.

## Port map

- Qwen3.6 MoE ForwardMoeHidden: vLLM `qwen3_5.py` (return_hidden branch,
  already in `DenseForwardLayers` at `qwen3_5.cpp:9835,9993-9995`) →
  `src/vllm/model_executor/models/qwen3_5.cpp` (add `ForwardMoeHidden`
  alongside existing `ForwardDenseHidden` at `:9424`).
- Multimodal vision tower: Qwen3-VL pattern (`qwen3_vl_vision.cpp`,
  `qwen3_vl_registry.cpp`) → reused for xor's image processing (vision weights
  loaded from xor checkpoint, vision config from nested `vision_config`).
- Decision head + candidate readout: xor reference (SGLang oracle) →
  `src/vllm/model_executor/models/xor_registry.cpp` (new file).
- Forward+reverse evaluation + calibration: xor reference (SGLang oracle) →
  `src/vllm/model_executor/models/xor_registry.cpp` (new file).
- SystemOne dispatch: shared `src/vllm/entrypoints/openai/systemone.{h,cpp}`
  (reused from Laya/kev).
- C ABI: `vllm_decide` / `vllm_decide_free` in `include/vllm.h:1192-1197` /
  `src/capi/vllm_c.cpp` (ABI v29, already exists). The architecture
  allowlist in `vllm_decide` (`src/capi/vllm_c.cpp:1711-1718`) currently
  accepts `KevModel`, `LayaModel`, `CuaS1Forms`. Add `"XorModel"` to
  this list and add an `XorInference` branch so xor is dispatched by
  engine architecture name — same pattern, same ABI, one more family.
  Similarly, `server_main.cpp` must route the `"XorModel"` architecture
  to the xor decision callback via `set_decision`.
- Registration: `src/vllm/model_executor/models/xor_registry.cpp` (new file,
  self-registers via `REGISTER_VLLM_MODEL`).
- Tests: `tests/vllm/models/test_xor.cpp` (new file).

## Tests to port

No upstream vLLM tests exist for xor (not in vLLM registry). Tests are
authored from the SGLang oracle and the xor model card:

- ForwardMoeHidden correctness: verify hidden-state extraction from the MoE
  backbone matches expected shapes and values for synthetic input (mirror
  kev's ForwardHidden golden tests).
- Multimodal image processing: verify vision tower forward + embedding merge
  for 1, 4, and 8 images.
- Decision head: verify single-token candidate readout produces correct
  probabilities for known inputs.
- Forward+reverse evaluation: verify that running options in forward and
  reverse order produces consistent results.
- Probability calibration: verify calibrated probabilities match SGLang oracle
  outputs within tolerance.
- Confidence formulas: choice `(max(p) - 1/K) / (1 - 1/K)`, score
  `1 - E|level - mode| / (L - 1)`, noul `p(true)` (confirmed against oracle).
- E2E: choice/score/noul through `/v1/systemone` via LocalAI HTTP server,
  cross-checked against SGLang oracle.
- E2E with images: choice/score/noul with 1-8 images through `/v1/systemone`.

## Dependencies

- The `/v1/systemone` API, `DecisionFn` callback, and `vllm_decide` C ABI
  (v29) — landed in the Laya/kev rows.
- The Qwen3.6 MoE weight loading infrastructure
  (`src/vllm/model_executor/models/qwen3_5_weights.cpp`, written against
  `Qwen/Qwen3.6-35B-A3B` and `nvidia/Qwen3.6-35B-A3B-NVFP4`).
- The Qwen3.6 MoE forward machinery (`Qwen3_5Model::Forward`,
  `RunMoeBlock` in `qwen3_5.cpp`, `qwen3_5_moe_block.h`).
- The `return_hidden` branch in `DenseForwardLayers` (`qwen3_5.cpp:9835`)
  — the precedent for hidden-state extraction.
- The Qwen3-VL vision tower (`qwen3_vl_vision.cpp`, `qwen3_vl_vision.h`,
  `qwen3_vl_registry.cpp`).
- The shared systemone helpers
  (`src/vllm/entrypoints/openai/systemone.{h,cpp}`).
- The SGLang oracle (`.agents/oracles/sglang.md`) for correctness cross-check.
- No new CUDA kernels — xor routes through existing `vt::` ops.

## Work breakdown

- Phase 1: Qwen3.6 MoE ForwardMoeHidden (hidden-state extraction for MoE
  variant) — TODO.
- Phase 2: Multimodal image processing (vision tower + embedding merge for up
  to 8 images) — TODO.
- Phase 3: Decision head + deterministic single-token candidate readout —
  TODO.
- Phase 4: Forward+reverse option-order evaluation + probability calibration
  — TODO.
- Phase 5: Registration, /v1/systemone dispatch, vllm_decide integration —
  TODO.
- Phase 6: GPU (CUDA) forward verification — TODO.

## Risks

- **Biggest risk: the Qwen3.6 MoE backbone in decision mode.** The
  text-generation MoE forward exists (`qwen3_5_moe.cpp`), but ForwardMoeHidden
  (hidden-state extraction) does not. The `return_hidden` branch exists in
  `DenseForwardLayers` (`qwen3_5.cpp:9835,9993-9995`) but has not been
  exercised for the MoE variant. The MoE block (`MoeBlock`, `RunMoeBlock`)
  produces hidden states, but the extraction path must be verified to produce
  correct results — the MoE expert routing and shared expert must run
  identically in feature-extraction and text-generation modes.
- **Multimodal adds complexity.** The Qwen3-VL vision tower exists but is
  wired for `Qwen3VLForConditionalGeneration` (a separate 4B dense model), not
  the Qwen3.6 MoE. The vision tower config, weight names, and embedding merge
  pattern must be verified against the xor checkpoint. Up to 8 images means the
  vision tower runs multiple times per request, increasing latency and memory.
  The MRoPE position computation must handle the multimodal case for the MoE
  backbone.
- **Forward+reverse evaluation doubles inference cost.** Each question
  requires two full forward passes (forward + reverse option order). For a 35B
  MoE, this is significant. Prefix caching of the shared state prefix could
  help, but is deferred.
- **Probability calibration formula.** The exact calibration method (how
  forward and reverse probabilities are combined) must be determined from the
  xor reference / SGLang oracle. An incorrect formula produces correct-looking
  but wrong probabilities.
- **Single-token candidate readout mechanism.** The exact readout (which token
  position, which vocabulary token, whether the LM head or a separate decision
  head is used) must be verified against the xor reference. An off-by-one in the
  candidate position silently produces wrong probabilities.
- **The 35B MoE checkpoint (~70 GB bf16) must be available** for testing.
- **SGLang oracle correctness cross-check.** The SGLang oracle must serve xor
  and produce reference outputs at the pinned revision. The greedy token-ID
  correctness cross-check (`SGLANG-ORACLE-CORRECT`) is `INVENTORIED` but may
  not have been run for xor specifically. If unavailable, the E2E parity gate
  uses distributional tolerance, not token-exact matching.

## Gates

- CPU-correct: all golden tests pass (ForwardMoeHidden, multimodal, decision
  head, forward+reverse, calibration, E2E).
- E2E parity: choice/score/noul outputs match SGLang oracle within tolerance,
  with and without images.
- Reachability: `/v1/systemone` endpoint serves xor model through
  `ModelRegistry::Forward` and `vllm_decide`.
- GPU build verification: owed (not pre-PR gate; stop condition is correct on
  CPU).

## Stop conditions

- CPU-correct + E2E parity + reachability = ready for PR.
- GPU build = owed.
- Prefix caching = future optimization.
- GGUF k-quants = owed.

## Owed

- GPU (CUDA) build verification.
- GGUF k-quant arm.
- Prefix caching optimization (shared state prefix across forward+reverse
  passes).
- SGLang oracle correctness cross-check: must verify SGLang serves xor at the
  pinned revision and produces reference outputs. If the greedy token-ID
  correctness cross-check (`SGLANG-ORACLE-CORRECT`) is not run for xor, the E2E
  parity gate uses distributional tolerance, not token-exact matching.

## Git integration

One pull request (repository default policy). Spec commit precedes
implementation commits in the same pull request.

## Weights

- Base: `Qwen/Qwen3.6-35B-A3B` — bf16, ~70 GB (35B total, ~3B activated).
- Post-trained: `juspay/xor` — fully merged BF16 weights (no adapter, no
  LoRA). Includes MoE backbone weights + vision tower weights + decision head
  weights + tokenizer.
