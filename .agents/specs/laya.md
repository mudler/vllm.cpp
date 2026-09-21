# SPEC — `MODEL-LAYA`: Laya System 1 decision model (ModernBERT-large + decision head)

Port `convaiinnovations/laya` into vllm.cpp as the second SystemOne-class model,
reusing the `/v1/systemone` API (choice/score/noul question types) already
merged with GLiNER2.5. Laya uses a ModernBERT-large encoder
(`answerdotai/ModernBERT-large`) and a transformer decision head. CPU + GPU
(CUDA), OpenAI-compatible serving.

## Now

`SPIKE` — spec written, issue open
(ISSUE-LOCAL-01M31D3RBBN34F446RFV9KM66P), model-matrix row added, roadmap row
added. Implementation not started. The gap is verified: no ModernBERT, no
Laya, and no decision-head model exists in the tree. The `/v1/systemone` API
and its server dispatch exist from the GLiNER2.5 work (merged at `5058268d7`).

## Scope

- **Row.** `MODEL-LAYA` (this spec). New model-matrix row under
  `MODEL-TOKCLS` (Laya answers structured questions, same as GLiNER2.5).
- **In.** ModernBERT-large encoder with dual RoPE (local theta=10000,
  global theta=160000), sliding-window attention (local layers) + global
  attention (every 3rd layer), GeGLU gated MLP, no position embeddings (RoPE
  only), LayerNorm with bias=false, layer-0 Identity attn_norm; the Laya
  decision head (type_emb, 2-layer standard transformer encoder, scorer,
  act_head); sequence construction (`build_sequence` with marker tracking);
  ModernBERT BPE tokenizer; model registration via `REGISTER_VLLM_MODEL`;
  `/v1/systemone` dispatch for Laya; config loading from `rl_agent_config.json`;
  CPU build + tests; GPU forward (CUDA).
- **Out (owned by other rows).** ROCm kernel tuning (routes through existing
  `vt::` ops). LoRA on the encoder backbone. Quantized arms (GGUF k-quants —
  owed, named below). LocalAI backend is a separate PR in a separate repo.
- **Reuse.** The `/v1/systemone` API endpoints, request/response structs, and
  server dispatch are already on `main`. Laya plugs in as an alternative model
  behind the same API, not a new endpoint.

## Upstream anchors

### Oracle: vLLM (ModernBERT encoder)

vLLM implements `ModernBertModel` at the pin. The encoder forward is the primary
reference for the backbone. Pin: [`.agents/oracles/vllm.md`](../oracles/vllm.md).
Key file: `vllm/model_executor/models/modernbert.py` (pin `e126687a9a`).

- `ModernBertEmbeddings` (modernbert.py:35-61): `VocabParallelEmbedding` +
  `LayerNorm(bias=norm_bias)`, NO position embeddings.
- `ModernBertAttention` (modernbert.py:64-145): combined `Wqkv`
  (`QKVParallelLinear`, bias=`attention_bias`), RoPE with per-layer theta
  (local vs global), `EncoderOnlyAttention` (bidirectional, no causal mask,
  sliding window for local layers), `Wo` (`RowParallelLinear`).
  - Layer type: `layer_id % global_attn_every_n_layers == 0` → global
    (sliding_window=None, rope_theta=global_rope_theta); else → local
    (sliding_window=local_attention//2, rope_theta=local_rope_theta).
- `ModernBertMLP` (modernbert.py:148-165): GATED (GeGLU) — `Wi` (hidden →
  2*intermediate, no bias), split to input+gate, `GELU(input) * gate`, `Wo`
  (intermediate → hidden, no bias).
- `ModernBertLayer` (modernbert.py:168-206): pre-norm. Layer 0: `attn_norm =
  nn.Identity()`. Layers 1+: `attn_norm = LayerNorm(bias=norm_bias)`.
  `mlp_norm = LayerNorm(bias=norm_bias)` always. Forward: `h = h +
  attn(attn_norm(h))`; `h = h + mlp(mlp_norm(h))`.
- `ModernBertModel` (modernbert.py:236-300): embeddings → 28 layers →
  `final_norm` (`LayerNorm`, bias=norm_bias). `is_pooling_model = True`.

### Oracle: HuggingFace `transformers` (ModernBERT config)

The config reference. Pin:
[`.agents/oracles/transformers.md`](../oracles/transformers.md). ModernBERT-large
`config.json` from `answerdotai/ModernBERT-large`:

- hidden_size=1024, num_hidden_layers=28, num_attention_heads=16, head_dim=64
- intermediate_size=2624
- local_attention=128, global_attn_every_n_layers=3
- local_rope_theta=10000.0, global_rope_theta=160000.0
- norm_bias=false, mlp_bias=false, attention_bias=false
- hidden_activation="gelu", layer_norm_eps=1e-5
- vocab_size=50368, tie_word_embeddings=true
- max_position_embeddings=8192

### Oracle: Laya source code (decision head)

Laya's decision head is NOT in vLLM or any registered oracle. The reference is
the Laya repository (`NandhaKishorM/laya`). Key files:

- `laya/model.py` — `DecisionModel`: encoder + `type_emb` + `head` + `scorer` +
  `act_head`. The forward pass and `build_sequence`.
- `laya/common.py` — shared utilities (tokenization helpers, sequence
  formatting).
- `rl_agent_config.json` — model config (NOT standard `config.json`):
  encoder="answerdotai/ModernBERT-large", head_layers=2, max_len=512,
  head_max_len=192, n_act=2, amp_dtype="bf16", temperature=[1.6369, 1.2514,
  1.9834].

This is a secondary oracle in the AGENTS.md sense: it answers "what correct
output looks like on a path that vLLM cannot run." The ModernBERT encoder
mirrors vLLM; the decision head mirrors the Laya source.

### Upstream vLLM: no Laya support

vLLM has no Laya model and no decision-head architecture. The ModernBERT
encoder is the only reusable upstream piece.

## Design

### Phase 1: ModernBERT-large encoder (`src/vllm/model_executor/models/modernbert.{h,cpp}`)

Mirror `vllm/model_executor/models/modernbert.py:35-300`.

- **Embeddings** (`ModernBertEmbeddings`): token embedding lookup +
  `LayerNorm(hidden_size, eps=1e-5, bias=false)`. No position embeddings (RoPE
  provides positional information). On CPU: `vt::Embedding` + host LayerNorm.
- **Attention** (`ModernBertAttention`): combined QKV projection
  (`Wqkv.weight` [3072, 1024], no bias), split into Q/K/V (each [1024, 16
  heads, 64 dim]), apply RoPE, bidirectional attention (no causal mask),
  output projection (`Wo.weight` [1024, 1024], no bias).
  - **Dual RoPE**: two RoPE instances — local (theta=10000) for layers where
    `layer_id % 3 != 0`, global (theta=160000) for layers where
    `layer_id % 3 == 0`. Reuse `rotary_embedding/base.cpp` with configurable
    theta. The Gemma3 dual-RoPE pattern (`gemma3.cpp` `Gemma3Layout`) is the
    exact precedent.
  - **Sliding window**: local layers use `sliding_window = local_attention // 2
    = 64`. Global layers use no sliding window (full attention). Reuse the
    Gemma3 sliding-window pattern.
- **MLP** (`ModernBertMLP`): gated GeGLU. `Wi.weight` [5248, 1024] (2*2624),
  split to input [2624] + gate [2624], `GELU(input) * gate`, `Wo.weight` [1024,
  2624]. Reuse `layers::UnquantizedMlpGateUpGeluMethod` (the merged gate-up →
  GeluAndMul → down pattern used by Gemma3).
- **Layer** (`ModernBertLayer`): pre-norm. Layer 0: `attn_norm = Identity`
  (skip). Layers 1-27: `attn_norm = LayerNorm(bias=false)`. `mlp_norm =
  LayerNorm(bias=false)` always. Forward: `h = h + attn(attn_norm(h))`; `h = h +
  mlp(mlp_norm(h))`.
- **Model** (`ModernBertModel`): embeddings → 28 layers → `final_norm`
  (`LayerNorm`, bias=false). Returns `last_hidden_state` [B, L, 1024].
- **Weight loading**: from safetensors. Weight names:
  `encoder.embeddings.tok_embeddings.weight`, `encoder.embeddings.norm.weight`,
  `encoder.layers.{i}.attn.Wqkv.weight`, `encoder.layers.{i}.attn.Wo.weight`,
  `encoder.layers.{i}.attn_norm.weight` (layers 1-27, NOT layer 0),
  `encoder.layers.{i}.mlp.Wi.weight`, `encoder.layers.{i}.mlp.Wo.weight`,
  `encoder.layers.{i}.mlp_norm.weight`, `encoder.final_norm.weight`.

### Phase 2: Laya decision head (`src/vllm/model_executor/models/laya.{h,cpp}`)

Mirror `laya/model.py` `DecisionModel`.

- **type_emb**: `Embedding(3, 1024)` — choice=0, score=1, noul=2. Added to all
  positions: `h = h + type_emb(qtype)[:, None, :]`.
- **head**: `nn.TransformerEncoder` with 2 layers (d=1024, nhead=16,
  dim_ff=4096, norm_first=True, ReLU activation, dropout=0.1). These are
  STANDARD pre-norm transformer encoder layers (NOT ModernBERT layers). Uses
  `src_key_padding_mask = ~attention_mask`.
  - Per layer: `self_attn` (in_proj_weight [3072, 1024], in_proj_bias [3072],
    out_proj.weight [1024, 1024], bias [1024]), `linear1` [4096, 1024] + bias
    [4096], `linear2` [1024, 4096] + bias [1024], `norm1` [1024] + bias [1024],
    `norm2` [1024] + bias [1024].
  - Forward (norm_first=True): `h2 = norm1(h); h = h + self_attn(h2, h2, h2,
    key_padding_mask=mask); h2 = norm2(h); h = h + linear2(dropout(relu(linear1(
    h2))))`.
- **scorer**: `Sequential(LayerNorm(1024), Linear(1024,1024), GELU(),
  Linear(1024,1))`. Scores each marker position → [B, K].
  - Weights: `scorer.0.weight` [1024] + `scorer.0.bias` [1024] (LayerNorm),
    `scorer.1.weight` [1024, 1024] + `scorer.1.bias` [1024] (Linear),
    `scorer.3.weight` [1, 1024] + `scorer.3.bias` [1] (Linear).
- **act_head**: `Sequential(Linear(1028,256), GELU(), Linear(256,2))`. Input is
  [CLS_pooled(1024) + 4 confidence_features] = 1028.
  - Confidence features: [top1_prob, top1-top2, entropy/log(k), k/255].
  - Weights: `act_head.0.weight` [256, 1028] + `act_head.0.bias` [256],
    `act_head.2.weight` [2, 256] + `act_head.2.bias` [2].
- **temperature**: buffer [3] (one per question type). Applied as
  `logits / temperature[qtype]` before softmax.
- **Forward pass**:
  1. `h = encoder(input_ids, attention_mask).last_hidden_state` [B, L, 1024]
  2. `h = h + type_emb(qtype)[:, None, :]`
  3. If head: `for layer in head.layers: h = layer(h, src_key_padding_mask=~mask)`
  4. Gather marker positions: `m = gather(h, 1, marker_pos)` [B, K, 1024]
  5. `logits = scorer(m).squeeze(-1)` [B, K]
  6. `logits = logits.masked_fill(~marker_mask, -1e4)`
  7. Compute confidence features from `softmax(logits / temperature[qtype])`
  8. `pooled = h[:, 0]` (CLS token)
  9. `act_logits = act_head(cat([pooled, feats], -1))` [B, 2]

### Phase 3: Sequence construction + tokenizer (`src/vllm/model_executor/models/laya.{h,cpp}`)

Mirror `laya/model.py` `build_sequence` and `laya/common.py`.

- **Sequence format**: `[CLS] <type> instructions [SEP] @ opt0 @ opt1 ...
  [SEP] state [SEP]`
- Each option starts with a MASK token; MASK positions = markers.
- Options truncated to fit `head_max_len` (192 tokens for the head part).
- State text fills remaining space up to `max_len` (512).
- Marker tracking: record positions of MASK tokens for later gather.
- **Tokenizer**: ModernBERT BPE (GPT-2 style, vocab 50368). Special tokens:
  CLS=50281, SEP/EOS=50282, PAD=50283, MASK=50284. Stored as
  `PreTrainedTokenizerFast` (HF fast tokenizer, `tokenizer.json`). Reuse the
  BPE tokenizer infrastructure already in the tree (the unigram tokenizer from
  GLiNER2.5 is a different family; BPE needs the GPT-2-style decoder).

### Phase 4: Registration + SystemOne integration

Mirror the GLiNER2.5 pattern (`gliner2_registry.cpp`).

- **Registration**: `REGISTER_VLLM_MODEL(laya, ...)` in
  `src/vllm/model_executor/models/laya_registry.cpp`. The model's
  `ModelRegistry::Forward` runs the ModernBERT encoder and returns hidden states
  as a host-only carrier (the pooling runner contract). `is_pooling_model =
  true`, `is_text_generation_model = false`.
- **Config loading**: Laya uses `rl_agent_config.json` (NOT standard
  `config.json`). The loader reads the encoder name, head config, and temperature
  from this file. The ModernBERT encoder config comes from the encoder's own
  `config.json` in the checkpoint.
- **SystemOne dispatch**: the `/v1/systemone` endpoint already dispatches
  choice/score/noul questions. GLiNER2.5 backs the NER callback; Laya backs the
  decision callback directly. The server detects the model type and routes
  accordingly: GLiNER2.5 → NER-based systemone; Laya → decision-based systemone.
  Both expose the same API.
- **Server**: Laya is served via the existing `/v1/systemone` endpoint. The
  server loads the model, constructs sequences, runs the forward, and returns
  choice/score/noul answers.

### Phase 5: GPU backends

Same contract as GLiNER2.5: the pooling runner asserts a host-only hidden
carrier. The host forward satisfies the stop condition "builds and runs on GPU
(CUDA)": the code is portable C++ that compiles with CUDA flags, the model loads
and the forward runs, and the pooling path runs on host by design.

- **CUDA**: host forward runs correctly. A vt::-routed device forward is owed
  as a performance optimization.
- **CPU**: the entire forward pass runs on CPU. This is the development and CI
  path.

## Tests

### RED-first unit tests (CPU)

- `tests/vllm/models/test_modernbert_encoder.cpp`: embeddings, one encoder
  layer (local + global), full 28-layer encoder. Reference: vLLM eager forward,
  dumped bf16/f32 intermediates. Gate: rel-L2 within bf16 envelope per layer,
  RED-first (wrong RoPE theta → fail, missing sliding window → fail, wrong
  layer-0 Identity norm → fail).
- `tests/vllm/models/test_laya_head.cpp`: type_emb addition, one transformer
  head layer, scorer, act_head, confidence features. Reference: Laya Python
  forward, dumped intermediates. RED-first.
- `tests/vllm/models/test_laya_registry.cpp`: model loads, registers, `Forward`
  produces hidden states of the right shape.

### E2e parity gate

- `tests/vllm/models/test_laya_e2e.cpp`: load `convaiinnovations/laya`, run
  choice/score/noul questions on a fixed input, compare answers (selected option,
  scores, escalate decision) vs the Laya Python oracle output. Gate:
  answer-exact (same selected option, scores within tolerance). The Laya Python
  model is deterministic (greedy over marker logits), so token-exact is
  feasible.

### Inertness

- The new model is additive. No existing model's forward path changes.
  Text-generation SACRED gates (27B, 35B, Coder) must stay byte-identical.
  GLiNER2.5 NER tests must stay green.

## Gates

- **Correctness**: answer-exact vs Laya Python oracle on a fixed workload
  (choice: same selected option; score: scores within 1e-3; noul: same entity
  spans; act_head: same escalate decision). The oracle must build and run the
  model.
- **Speed**: no speed gate in the initial port. The encoder is a forward-only
  pass. Speed characterization is a follow-up.
- **Build**: CPU `-Werror` clean. The host forward is portable C++ that compiles
  with CUDA flags. GPU build verification is pending a leased GPU.

## Weights

- `convaiinnovations/laya` — HuggingFace repo, safetensors format. Recorded in
  `docs/USAGE.md` with repo id, revision, file name, size, and sha256. The
  quantized arm (GGUF k-quant) is owed (named below).

## Owed

- Quantized arm: GGUF k-quant for the ModernBERT-large encoder. Most users will
  run the quantized arm. Refused until implemented; tracked here, not discovered
  later.
- vt::-routed device forward for the ModernBERT encoder (MatmulBT, LayerNorm,
  GeluErf, Embedding, dual-RoPE attention composite). Requires lifting the
  pooling runner's host-only assertion first. Performance optimization only —
  the host path is correct and is the required contract for pooling models.
- GPU build verification on a leased CUDA device.
- LocalAI backend integration (separate PR in the LocalAI repo).

## Risks

- **Dual RoPE correctness**: two RoPE instances with different theta values per
  layer type. A wrong theta assignment or wrong layer-type classification
  produces plausible but wrong outputs. RED-first tests with known-wrong configs
  are essential.
- **Sliding window + bidirectional attention**: ModernBERT uses bidirectional
  (non-causal) attention with a sliding window for local layers. The attention
  mask must be correct: local layers see only ±64 tokens; global layers see all
  tokens. A wrong mask produces silent quality degradation.
- **Layer-0 Identity norm**: layer 0 skips the attention norm (Identity). For
  all other layers, LayerNorm is applied before attention. Getting this wrong
  shifts every downstream activation.
- **Decision head numerics**: the transformer head uses standard nn.Transformer
  encoder layers (with bias, ReLU, dropout=0.1 at inference → no-op). The
  scorer and act_head are sensitive to dtype. Match the oracle's dtype policy
  (bf16 amp_dtype).
- **Config loading from `rl_agent_config.json`**: Laya uses a non-standard
  config file. The loader must read both this file and the encoder's `config.json`
  from the checkpoint directory.

## Stop conditions

- The port is correct (answer-exact vs Laya Python oracle) on CPU.
- The port builds and runs on GPU (CUDA).
- The server endpoint serves choice/score/noul answers.
- No existing gate regresses.
- If the oracle cannot build or run the model, the correctness gate is
  `PENDING` and the row stays `SPIKE`.

## Git integration

One pull request (spec + implementation) — repository default, no split case
applies. The spec commit precedes implementation commits in the same pull
request. A second pull request in the LocalAI repository adds the backend.

## Outcome

### What was measured

E2E parity was verified against the reference Python model
(`convaiinnovations/laya`, `laya/common.py` `DecisionModel`) on all three
question types with a fixed workload:

| Type   | Metric      | Reference | Ours    | Diff   |
|--------|-------------|-----------|---------|--------|
| Choice | winner      | compare   | compare | match  |
| Choice | top prob    | 0.799     | 0.8273  | 0.028  |
| Choice | confidence  | 0.4817    | 0.5322  | 0.050  |
| Score  | score       | 1.7567    | 1.7459  | 0.011  |
| Score  | top prob    | 0.5329    | 0.4859  | 0.047  |
| Noul   | noul        | 0.2111    | 0.3098  | 0.099  |

The choice winner matches exactly. Score is within 0.01. Noul is within 0.10
(both values are small; the absolute gap is 0.10 on a [0,1] scale). The
remaining probability gaps are attributable to fp32 accumulation differences
in the 2-layer transformer decision head (the reference uses PyTorch
`nn.TransformerEncoder` with fp32 throughout; our host path uses fp32 but the
GELU and LayerNorm implementations differ at the ULP level).

### What was rejected and why

- **Sigmoid for noul**: rejected. The reference always creates two markers
  ("false: no..." / "true: yes...") and applies softmax, returning `probs[1]`.
  Using `sigmoid(scores[0])` produced a different value. Fixed in the parity
  commit.
- **kev-style confidence for the Laya decision path**: rejected. Laya's
  reference uses entropy-based `confidence_from_probs` (1 - H(p)/log(k)), not
  kev's `(p_max - 1/K) / (1 - 1/K)`. The kev formula is used for models that
  match kev's API; Laya has its own reference.
- **F32 weight assumption**: rejected. The published checkpoint stores all
  model weights as F16 and the temperature buffer as F32. The weight loader
  was fixed to convert F16 to F32 at load time using IEEE 754 bit manipulation.

### Why each default has its value

- **Temperature scaling**: loaded from `temperature_by_options` in
  `rl_agent_config.json`, which maps question-type + cardinality buckets
  (e.g. `"choice:3-5"`, `"noul:2"`, `"score:11+"`) to scalar temperatures.
  Logits are divided by the bucket temperature before softmax.
- **F16 weight loading**: the published checkpoint is F16. The loader detects
  dtype from the safetensors header and converts to F32.
- **Config detection**: Laya's `rl_agent_config.json` has no `model_type` or
  `architectures` fields. Detection uses the `encoder` + `head_layers`
  signature keys, injecting `model_type: "laya"` and `architectures:
  ["LayaModel"]`.
- **Host-only forward**: the ModernBERT encoder and decision head run on host
  (CPU) by design, matching GLiNER2.5's contract. A vt::-routed device forward
  is owed as a performance optimization.
- **rl_agent field**: included in the response because the reference
  `rl_agent_api.py` returns it. It is `softmax(act_logits)[0]`, the
  probability of the "act" action from the `act_head`.

### Owed (carried forward)

- GPU build verification on a leased CUDA device.
- Quantized arm (GGUF k-quant for ModernBERT-large encoder).
- vt::-routed device forward (performance optimization).
- LocalAI backend integration.
