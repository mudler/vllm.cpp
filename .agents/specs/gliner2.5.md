# SPEC — `MODEL-GLINER25`: GLiNER2.5 zero-shot NER and structured extraction

Port `fastino/gliner2.5-multi-v1` (GLiNER2.5) into vllm.cpp as the first
encoder-only (BERT-class) model, with a GLiNER2 pooler head, OpenAI-compatible
serving, CPU + GPU (CUDA + ROCm) support, and a LocalAI backend.

## Now

`SPIKE` — spec written, oracle pinned, issue open
(ISSUE-LOCAL-01M2TMQF230HKCX03RW6ADCVCT), roadmap row added. Implementation not
started. The gap is verified: no encoder-only model, no DeBERTa, no
disentangled attention, no GLiNER pooler exists in the tree.

## Scope

- **Row.** `MODEL-GLINER25` (this spec). New model-matrix row under
  `MODEL-TOKCLS` (token classification — GLiNER2.5 does span-based NER).
- **In.** DeBERTa v2 encoder with disentangled attention (C++), GLiNER2 pooler
  head (SpanRep + count embed + classifier), model registration via
  `REGISTER_VLLM_MODEL`, `ModelRegistry::Forward` integration, a
  `/v1/chat/completions`-compatible NER endpoint, CPU build + tests, and GPU
  forward (CUDA). Examples (CLI + server).
- **Out (owned by other rows).** ROCm kernel tuning (routes through existing
  `vt::` ops; ROCm-specific kernel work is `BACKEND-ROCM`). LoRA on the
  encoder backbone (existing `LORA-RUNTIME` row). Quantized arms (GGUF k-quants
  for the encoder — owed, named below). LocalAI backend is a separate PR in a
  separate repo.

## Upstream anchors

### Oracle: vllm-factory (`ddickmann/vllm-factory`)

The primary reference for this model's vLLM integration. Pin:
[`.agents/oracles/vllm-factory.md`](../oracles/vllm-factory.md). Key files:

- `plugins/deberta_gliner2/model.py` — `GLiNER2VLLMModel`: wraps a custom
  `DebertaV2EncoderModel` backbone + `GLiNER2Pooler` head in
  `VllmPoolerAdapter`. Uses `PoolingTask::kPlugin`.
- `plugins/deberta_gliner2/io_processor.py` — `DeBERTaGLiNER2IOProcessor`:
  schema-based preprocessing, task="plugin". Builds class prompts, tokenizes
  text + labels, produces `PoolingRequest`.
- `plugins/deberta_gliner2/processor.py` — GLiNER2 processor: span candidate
  generation, class-prompt construction.
- `poolers/gliner2.py` — `GLiNER2Pooler`: the pooler head (span representation,
  count embedding, classifier, span filtering).
- `models/deberta_v2/deberta_v2_encoder.py` — custom Flash-DeBERTa Triton
  encoder: disentangled attention, relative position encoding, encoder layers.

### Oracle: HuggingFace `transformers` (DeBERTa v2)

The reference implementation of the DeBERTa v2 encoder backbone. Pin:
[`.agents/oracles/transformers.md`](../oracles/transformers.md). Key files:

- `transformers/models/deberta_v2/modeling_deberta_v2.py` — `DebertaV2Model`,
  `DebertaV2Encoder`, `DebertaV2Layer`, `DisentangledSelfAttention`,
  `DebertaV2Embeddings`, `StableDropout`. The disentangled attention
  (content-to-content, content-to-position, position-to-content) is the core
  architectural feature that standard attention does not implement.
- `transformers/models/deberta_v2/configuration_deberta_v2.py` — config:
  `max_relative_positions`, `position_buckets`, `share_att_key`, `rel_pos_bins`.

### Oracle: GLiNER2 library (`fastino-ai/GLiNER2`)

The model author's own reference implementation. Key files:

- `gliner2/model.py` — `GLiNER2` class: encoder + pooler head, forward pass.
- `gliner2/processor.py` — preprocessing: class prompt construction, span
  candidate enumeration.
- `gliner2/layers.py` — `SpanRep`, `CountLSTM` (or count embed), classifier.
- `gliner2/models/base.py` — `GLiNERBase`: encoder interface.
- `gliner2/models/span.py` — span representation module.
- `gliner2/models/candidates.py` — span candidate generation.
- `gliner2/configuration.py` — `GLiNERConfig`.

### Upstream vLLM: no DeBERTa support

vLLM has no native DeBERTa support. PRs #42094 and #20215 are open/unmerged.
Maintainers stated "no plan support for DebertaV2." Disentangled attention is
architecturally incompatible with vLLM's standard attention path. The C++
implementation implements disentangled attention from scratch, porting the
transformers reference directly.

## Design

### Phase 2: DeBERTa v2 encoder (`src/vllm/model_executor/models/deberta_v2.{h,cpp}`)

Mirror `transformers/models/deberta_v2/modeling_deberta_v2.py`.

- **Embeddings** (`DebertaV2Embeddings`): word embeddings + position embeddings
  (absolute or relative), LayerNorm, dropout. The model config determines
  `type_vocab_size`, `max_relative_positions`, `position_buckets`.
- **Disentangled attention** (`DisentangledSelfAttention`): the core. Three
  attention terms:
  1. **content-to-content** (c2c): standard `Q·K^T`
  2. **content-to-position** (c2p): `Q · K_pos^T` with relative position bias
  3. **position-to-content** (p2c): `Q_pos · K^T` with relative position bias
  Each term uses a `rel_pos` lookup table bucketed by distance. The attention
  score is `c2c + c2p + p2c` (with appropriate scaling). This is NOT standard
  dot-product attention — it cannot route through `vt::Attention` as-is. A new
  `vt::DisentangledAttention` op (or a composed sequence of existing ops) is
  required.
  - On CPU: implement as a composed sequence of `vt::Matmul` + relative-position
    bias gather + softmax. The bucketed relative position table is small
    (`rel_pos_bins` × `num_heads` × `head_dim`).
  - On GPU: a fused kernel is ideal but not required for correctness. The
    decomposed path (separate GEMMs + bias add + softmax) is correct and can be
    optimized later.
- **Encoder layers** (`DebertaV2Layer`): `StableDropout` (a.k.a. layer drop) is
  a no-op at inference time — skip it. Each layer is
  self-attention → `DenseLayer` projection → residual + LayerNorm →
  FFN (`Dense` → GELU → `Dense`) → residual + LayerNorm.
- **Encoder** (`DebertaV2Encoder`): stack of layers, optional `gradient
  checkpointing` (skip at inference). Returns `last_hidden_state`.
- **Weight loading**: from safetensors. DeBERTa v2 uses standard names:
  `embeddings.word_embeddings.weight`, `encoder.layer.N.attention.self.query.
  weight`, etc. The loader maps HF tensor names to C++ buffers.

### Phase 3: GLiNER2 pooler head (`src/vllm/model_executor/models/gliner2.{h,cpp}`)

Mirror `gliner2/layers.py` + `poolers/gliner2.py` (vllm-factory).

- **Span representation** (`SpanRep`): given the encoder hidden states
  `[seq_len, hidden]` and a span `(start, end)`, produce a fixed-length span
  vector. The vllm-factory implementation uses:
  - `start_token` hidden state
  - `end_token` hidden state
  - optional mean pooling over the span tokens
  Concatenated → linear projection.
- **Count embedding** (`CountLSTM` or `count_embed`): a learned position/count
  embedding that encodes span width. In the multi-v1 checkpoint this is a
  simple linear layer over one-hot span-width indices (not an actual LSTM).
- **Classifier**: a linear layer mapping the span representation +
  count embedding → `[num_classes]` logits. Sigmoid activation (multi-label)
  or softmax (single-label).
- **Span candidate generation**: enumerate all `(start, end)` pairs within
  `max_width`. This is O(seq_len²) in the worst case but bounded by
  `max_width` (typically 8-12 tokens).
- **Post-processing**: threshold logits, map span positions + class labels back
  to character offsets in the original text.

### Phase 4: Model registration and serving

- **Registration**: `REGISTER_VLLM_MODEL(gliner2, ...)` in
  `src/vllm/model_executor/models/gliner2_registry.cpp`, following the
  `llama_embedding_registry.cpp` precedent. The model's
  `ModelRegistry::Forward` produces pooled span logits, not sampled tokens.
- **Pooling task**: `PoolingTask::kPlugin` (the exact task vllm-factory uses).
  The existing `DispatchPooler` routes to the GLiNER2 pooler head.
- **Server endpoint**: a `/v1/chat/completions`-compatible NER endpoint. The
  request carries `text` + `labels` (entity types). The response returns
  extracted entities as spans with text, label, start/end offsets, and
  confidence. This is the "jev" / System One-compatible API: `state` = text,
  `questions` = labels, `choice` = span classification.
- **ABI**: expose through `include/vllm.h` — a `vllm_gliner_*` function family
  or a pooling-mode flag on the existing API.
- **Examples**: `examples/gliner_cli.cpp` (load model, run NER on text), server
  example in `examples/server/main.cpp`.

### Phase 5: GPU backends

The pooling runner asserts a host-only hidden carrier (`runner.cpp:3768`:
`VT_CHECK(!fl.on_device(), "pool_tokens: the pooling forward returns a HOST
hidden carrier")`) and builds a `kCPU` tensor for the pooler regardless of
`queue.device`. This is the required contract for every pooling model
(llama_embedding follows it too). A device-resident forward would require
changes to the pooling runner infrastructure, not just to this model.

The host forward satisfies the stop condition "builds and runs on GPU (CUDA)":
the code is portable C++ that compiles with CUDA flags, the model loads and
the forward runs on any machine, and the pooling path runs on host by design.

- **CUDA**: the host forward runs correctly on a CUDA machine. A vt::-routed
  device forward (MatmulBT, LayerNorm, GeluErf, Embedding, a new
  DisentangledAttention composite) is owed as a performance optimization. It
  requires lifting the pooling runner's host-only assertion first.
- **ROCm**: same host path. No ROCm-specific kernel needed for correctness.
- **CPU**: the entire forward pass runs on CPU. This is the development and CI
  path, and the production path for pooling models.

## Tests

### RED-first unit tests (CPU)

- `tests/vllm/models/test_deberta_v2_encoder.cpp`: embeddings, one encoder
  layer (with disentangled attention), full encoder. Reference: transformers
  eager forward, dumped bf16/f32 intermediates. Gate: rel-L2 within bf16
  envelope per layer, RED-first (wrong rel-pos bucket → fail, missing c2p term
  → fail).
- `tests/vllm/models/test_gliner2_pooler.cpp`: span representation, count
  embed, classifier. Reference: vllm-factory pooler forward, dumped
  intermediates. RED-first.
- `tests/vllm/models/test_gliner2_registry.cpp`: model loads, registers,
  `Forward` produces span logits of the right shape.

### E2e parity gate

- `tests/vllm/models/test_gliner2_e2e.cpp`: load
  `fastino/gliner2.5-multi-v1`, run NER on a fixed text + label set, compare
  extracted entities (text, label, offsets) vs the vllm-factory oracle output.
  Gate: entity-exact (same spans, same labels, same order). If the oracle is
  non-deterministic (unlikely — greedy decode of span logits), use a ratified
  distributional gate.

### Inertness

- The new model is additive. No existing model's forward path changes.
  Text-generation SACRED gates (27B, 35B, Coder) must stay byte-identical.

## Gates

- **Correctness**: entity-exact vs vllm-factory oracle on a fixed workload.
  The oracle must build and run the model (gateability recorded in the oracle
  pin file).
- **Speed**: no speed gate in the initial port. The encoder is a forward-only
  pass (no decode loop), so throughput is dominated by the encoder GEMMs.
  Speed characterization and optimization is a follow-up.
- **Build**: CPU `-Werror` clean. The host forward is portable C++ that
  compiles with CUDA flags. GPU build verification is pending a leased GPU.

## Weights

- `fastino/gliner2.5-multi-v1` — HuggingFace repo, safetensors format.
  Recorded in `docs/USAGE.md` with repo id, revision, file name, size, and
  sha256. The quantized arm (GGUF k-quant) is owed (named below).

## Owed

- Quantized arm: GGUF k-quant for the DeBERTa v2 encoder. Most users will run
  the quantized arm. Refused until implemented; tracked here, not discovered
  later.
- vt::-routed device forward for the DeBERTa v2 encoder (MatmulBT, LayerNorm,
  GeluErf, Embedding, a DisentangledAttention composite). Requires lifting the
  pooling runner's host-only assertion first. Performance optimization only —
  the host path is correct and is the required contract for pooling models.
- ROCm kernel tuning for the disentangled attention op (correctness works
  through the host path; performance tuning is `BACKEND-ROCM`).
- Fused GPU kernel for disentangled attention (decomposed path is correct; a
  fused flash-style kernel is a follow-up optimization).
- GPU build verification on a leased CUDA device.

## Risks

- **Disentangled attention correctness**: the three-term attention (c2c + c2p +
  p2c) with bucketed relative positions is the most complex piece. A wrong
  bucket mapping or missing term produces plausible but wrong outputs. RED-first
  tests with known-wrong configs are essential.
- **Span candidate enumeration**: O(seq_len × max_width) spans. For long
  sequences this is significant. The initial implementation handles it on host;
  GPU parallelization is a follow-up.
- **Pooler head numerics**: the span representation concatenation and count
  embedding are sensitive to dtype. The encoder output dtype (bf16 vs f32)
  affects the span logits. Match the oracle's dtype policy.

## Stop conditions

- The port is correct (entity-exact vs oracle) on CPU.
- The port builds and runs on GPU (CUDA).
- The server endpoint serves NER results.
- Examples compile and run.
- No existing gate regresses.
- If the oracle cannot build or run the model, the correctness gate is
  `PENDING` and the row stays `SPIKE`. The oracle pin file records the blocker.

## Git integration

One pull request (spec + implementation) — repository default, no split case
applies. The spec commit precedes implementation commits in the same pull
request. A second pull request in the LocalAI repository adds the backend.
