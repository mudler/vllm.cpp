# SPEC — `MODEL-CUA-S1-FORMS`: cua-s1-forms option scorer (byte-level transformer + cross-attention)

Port `cua-ai/cua-s1-forms` into vllm.cpp as the third SystemOne-class model.
Unlike GLiNER2.5 and Laya, which answer choice/score/noul questions through
`/v1/systemone`, cua-s1-forms takes a context string and a list of option
strings and returns one probability per option in a single forward pass. The
model is tiny: 706,048 trainable parameters, 2.8 MB checkpoint. CPU + GPU
(CUDA), OpenAI-compatible serving through a new `/v1/score` endpoint.

## Now

`SPIKE` — spec written, issue open
(ISSUE-LOCAL-01M32WZMKVRBP4PH21KPHEDS3V), model-matrix row added, roadmap row
added, oracle pinned. Implementation not started. The gap is verified: no
byte-level embedding model, no `TinyTransformerScorer`, and no option-scoring
endpoint exists in the tree.

## Scope

- **Row.** `MODEL-CUA-S1-FORMS` (this spec). New model-matrix row under
  `MODEL-SEQCLS` (the model scores a sequence of options against a context).
- **In.** `TinyTransformerScorer` (byte-level embedding, 2-layer transformer
  encoder for context, 1-layer transformer encoder for options, `AttentionHead`
  cross-attention scorer); `ByteCollator` (UTF-8 byte tokenization, padding,
  masking); model registration via `REGISTER_VLLM_MODEL`; safetensors weight
  loading from `cua-s1-forms.safetensors`; config loading from
  `cua-s1-forms.json`; a `/v1/score` endpoint (context + options to
  probabilities); CPU build + tests; GPU forward (CUDA).
- **Out (owned by other rows).** ROCm kernel tuning (routes through existing
  `vt::` ops). Quantized arms (GGUF k-quants — owed, named below). The
  downstream `Planner`, `Driver`, `RuntimeService`, and MCP server from the
  upstream source are not ported; vllm.cpp exposes the scoring primitive, not
  the GUI automation pipeline. LocalAI backend is a separate PR in a separate
  repo.
- **Reuse.** The existing safetensors reader
  (`include/vllm/entrypoints/model_loader/safetensors_reader.h`), the
  `HfConfig` JSON parser (`include/vllm/transformers_utils/hf_config.h`), and
  the server framework (`api_server.h`). The `/v1/score` endpoint is new; it
  does not reuse `/v1/systemone` because the input/output contract is different.

## Upstream anchors

### Oracle: trycua/cua (cua-s1-forms source)

The primary reference for this model. Pin:
[`.agents/oracles/cua.md`](../oracles/cua.md). Key files at
`libs/cua-s1/python/src/cua_s1/`:

- `model.py:166-236` — `TinyTransformerScorer`: byte-level `nn.Embedding(257,
  width, padding_idx=0)`, position `nn.Embedding(max(context_tokens,
  option_tokens), width)`, 2-layer `nn.TransformerEncoder` (context), 1-layer
  `nn.TransformerEncoder` (options), `AttentionHead`. The forward pass
  embeds context and options separately, encodes each, mean-pools option
  tokens, then scores through cross-attention.
- `model.py:102-136` — `AttentionHead`: `LayerNorm` on context and options,
  `Linear(width, rank, bias=False)` for query/key/value, scaled dot-product
  attention (options attend to context), then `(query * attended).sum(-1) /
  sqrt(rank)` produces one logit per option. Masked-fill with `finfo.min` for
  padding.
- `model.py:52-99` — `ByteCollator`: UTF-8 byte tokenization (`byte + 1`,
  pad=0), context/option tensor construction with masks.
- `model.py:48-49` — `_byte_ids`: `[byte + 1 for byte in text.encode("utf-8",
  errors="replace")[:length]]`.
- `model.py:208-210` — `_embed`: `embedding(ids) + position(arange(ids.shape[-1]))`.
- `model.py:212-236` — `forward`: context through encoder with
  `src_key_padding_mask`, options flattened and encoded separately, mean-pooled
  with mask weights, then `AttentionHead.forward`.
- `model.py:257-282` — `make_system`: constructs `TinyTransformerScorer` when
  `encoder == "tinyx"`. Validates `width`, `rank`, `context_tokens`,
  `option_tokens`, `layers`, `heads`.
- `checkpoint.py:40-55` — `resolve_checkpoint_paths`: `.safetensors` and `.json`
  sidecar convention.
- `checkpoint.py:58-122` — `save_checkpoint_files`: safetensors + JSON with
  `format`, `format_version`, `state_signature`, `config`, `metadata`.
- `checkpoint.py:125-181` — `load_checkpoint_files`: validates format name
  (`cua-s1`), version (1), state signature, loads safetensors state dict.
- `schema.py:53-64` — `render_context`: the context string format for GUI
  form-filling (`TASK fill the form ...`, `FORM ...`, `ELEMENT ...`).
- `schema.py:71-73` — `render_options`: entity options plus fixed actions
  (`check`, `click`, `skip`).
- `planner.py:60-155` — `Planner`: validates decisions, not part of the model
  port but documents the scoring contract.

### HuggingFace: `cua-ai/cua-s1-forms`

The published checkpoint. Files:
- `cua-s1-forms.safetensors` — weights (2.8 MB)
- `cua-s1-forms.json` — config + metadata
- `cua-s1-forms.pt` — legacy PyTorch checkpoint (not used)

Config (from `cua-s1-forms.json`):
```json
{
  "config": {
    "context_tokens": 224,
    "encoder": "tinyx",
    "heads": 4,
    "hf_model": "Qwen/Qwen2.5-0.5B",
    "layers": 2,
    "option_tokens": 96,
    "rank": 128,
    "width": 128
  },
  "format": "cua-s1",
  "format_version": 1
}
```

The `hf_model` field is metadata; the model does not use Qwen weights. All
weights are in the safetensors file.

### Upstream vLLM: no cua-s1-forms support

vLLM has no byte-level embedding model and no option-scoring architecture.
This is a from-scratch port of the upstream Python model.

## Design

### Architecture: `TinyTransformerScorer`

Parameters (from config):
- `width` = 128 (model hidden dimension)
- `rank` = 128 (attention head rank)
- `context_tokens` = 224 (max context bytes)
- `option_tokens` = 96 (max option bytes)
- `layers` = 2 (context encoder layers)
- `heads` = 4 (attention heads)
- `dim_ff` = `width * 4` = 512 (feed-forward dimension)
- `vocab_size` = 257 (byte + 1, pad = 0)

Modules and weight tensor names (PyTorch `named_parameters`):

1. `embedding.weight` — `(257, 128)`, `padding_idx=0`
2. `position.weight` — `(224, 128)` (max of context_tokens and option_tokens)
3. Context encoder (`encoder.layers.{0,1}`):
   - `self_attn.in_proj_weight` — `(384, 128)` (packed QKV, 3 * 128)
   - `self_attn.in_proj_bias` — `(384,)`
   - `self_attn.out_proj.weight` — `(128, 128)`
   - `self_attn.out_proj.bias` — `(128,)`
   - `linear1.weight` — `(512, 128)`, `linear1.bias` — `(512,)`
   - `linear2.weight` — `(128, 512)`, `linear2.bias` — `(128,)`
   - `norm1.weight` — `(128,)`, `norm1.bias` — `(128,)`
   - `norm2.weight` — `(128,)`, `norm2.bias` — `(128,)`
4. Option encoder (`option_encoder.layers.0`): same structure, 1 layer
5. Attention head (`head`):
   - `context_norm.weight` — `(128,)`, `context_norm.bias` — `(128,)`
   - `option_norm.weight` — `(128,)`, `option_norm.bias` — `(128,)`
   - `query.weight` — `(128, 128)`, no bias
   - `key.weight` — `(128, 128)`, no bias
   - `value.weight` — `(128, 128)`, no bias

All weights are `float32`. The checkpoint stores them as such; no dtype
conversion is needed at load time.

### Forward pass

Input: `context_ids` `(batch, ctx_len)`, `context_mask` `(batch, ctx_len)`,
`option_ids` `(batch, n_opt, opt_len)`, `option_token_mask` `(batch, n_opt,
opt_len)`, `option_mask` `(batch, n_opt)`.

1. **Context encoding**:
   - Embed: `embedding(context_ids) + position(arange(ctx_len))`
   - Safe mask: force `context_mask[:, 0] = True` (avoid all-padding rows)
   - Encode: `encoder(embedded, src_key_padding_mask=~safe_context_mask)`
   - Output: `context` `(batch, ctx_len, 128)`

2. **Option encoding**:
   - Flatten: `option_ids.reshape(batch * n_opt, opt_len)`
   - Embed: `embedding(flat_ids) + position(arange(opt_len))`
   - Safe mask: force `flat_mask[:, 0] = True`
   - Encode: `option_encoder(embedded, src_key_padding_mask=~safe_mask)`
   - Mean-pool: `(hidden * mask_weights).sum(1) / mask_weights.sum(1).clamp_min(1)`
   - Reshape: `pooled.reshape(batch, n_opt, 128)`

3. **AttentionHead scoring** (`model.py:116-136`):
   - Normalize: `context = context_norm(context.float())`, `options =
     option_norm(options.float())`
   - Project: `query = query_linear(options)` `(batch, n_opt, rank)`,
     `key = key_linear(context)` `(batch, ctx_len, rank)`, `value =
     value_linear(context)` `(batch, ctx_len, rank)`
   - Attention scores: `scores = einsum("bnr,blr->bnl", query, key) /
     sqrt(rank)` `(batch, n_opt, ctx_len)`
   - Mask: `scores.masked_fill(~context_mask[:, None, :], finfo.min)`
   - Attend: `attended = einsum("bnl,blr->bnr", scores.softmax(-1), value)`
     `(batch, n_opt, rank)`
   - Logits: `logits = (query * attended).sum(-1) / sqrt(rank)` `(batch, n_opt)`
   - Mask: `logits.masked_fill(~option_mask, finfo.min)`

4. **Output**: `softmax(logits)` over the live option dimension gives
   probabilities. The argmax is the winner.

### ByteCollator

`_byte_ids(text, length)` (model.py:48-49):
```
[byte + 1 for byte in text.encode("utf-8", errors="replace")[:length]]
```
- Byte values 0-255 map to token IDs 1-256. Token 0 is padding.
- Context truncated to `context_tokens` (224) bytes.
- Each option truncated to `option_tokens` (96) bytes.

Batch construction (`_tensor_batch`, model.py:71-99):
- `context_ids` `(batch, max_ctx_len)`, padded with 0
- `context_mask` = `context_ids != 0`
- `option_ids` `(batch, max_n_opt, max_opt_len)`, padded with 0
- `option_token_mask` = `option_ids != 0`
- `option_mask` `(batch, max_n_opt)`, True for valid options

### API endpoint: `/v1/score`

New endpoint, not `/v1/systemone`. The contract is simpler.

Request:
```json
{
  "context": "TASK fill the form from the document, then submit\nFORM ...",
  "options": ["fill Name: John", "fill Phone: 555-0100", "check", "click", "skip"]
}
```

Response:
```json
{
  "probabilities": [0.05, 0.72, 0.10, 0.08, 0.05],
  "winner": 1,
  "confidence": 0.72
}
```

- `probabilities`: softmax output, one per option
- `winner`: argmax index
- `confidence`: `max(probabilities)` (the winning option's probability)

The endpoint accepts a single context + options pair (batch = 1). The model
is small enough that batch processing is not a concern for the initial port.

### Config loading

The checkpoint sidecar `cua-s1-forms.json` wraps the config in a
`{format, format_version, state_signature, config, metadata}` envelope. The
loader reads the `config` object for `width`, `rank`, `context_tokens`,
`option_tokens`, `layers`, `heads`, and `encoder`. It validates `format ==
"cua-s1"` and `format_version == 1`.

### Weight loading

Safetensors file `cua-s1-forms.safetensors` contains the state dict with
standard PyTorch parameter names. The loader maps each tensor by name to the
C++ model's weight buffers. All tensors are `float32`.

### Registration

Register through `REGISTER_VLLM_MODEL` with a `ModelInfo` entry. The model
factory reads the sidecar JSON config, constructs the `TinyTransformerScorer`,
and loads weights from the safetensors file. The server dispatches `/v1/score`
requests to the model's forward function.

### File structure (mirrors vLLM)

- `include/vllm/model_executor/models/cua_s1.h` — model header
- `src/vllm/model_executor/models/cua_s1.cpp` — model implementation
  (embedding, transformer encoder, attention head, forward pass)
- `include/vllm/model_executor/models/cua_s1_collator.h` — `ByteCollator`
  (byte tokenization, tensor construction)
- `src/vllm/model_executor/models/cua_s1_weights.cpp` — weight loader
- `src/vllm/model_executor/models/cua_s1_registry.cpp` — registration TU
  (ModelInfo, factory, forward dispatch)
- `include/vllm/model_executor/models/cua_s1_inference.h` — inference entry
  point (`CuaS1ScoreResult`, `CuaS1Inference()`)
- `scripts/gen-cua-s1-goldens.py` — PyTorch golden generator
- `tests/vllm/models/cua_s1_goldens.inc` — generated goldens
- `tests/vllm/models/test_cua_s1.cpp` — unit tests

## Phases

### Phase 1: `TinyTransformerScorer` model

Port the model architecture from `model.py:166-236`. Implement in C++:
- `ByteEmbedding` (embedding + position, `_embed` at model.py:208-210)
- `TransformerEncoderLayer` (pre-norm, batch_first, `norm_first=True`):
  `self_attn` (multi-head attention with packed QKV), `linear1`/`linear2`
  (feed-forward with ReLU activation), `norm1`/`norm2` (LayerNorm). PyTorch
  `nn.TransformerEncoderLayer` uses ReLU by default.
- `TransformerEncoder` (stack of layers)
- `AttentionHead` (model.py:102-136): context/option LayerNorm, query/key/value
  linear projections, scaled dot-product cross-attention, logit computation
- Forward pass: context encoding, option encoding + mean-pooling, attention
  head scoring, softmax

Test: golden comparison against PyTorch `TinyTransformerScorer` with the
published checkpoint. Verify logits and probabilities match to `1e-5`.

### Phase 2: `ByteCollator` + inference

Port `ByteCollator` (model.py:52-99) and `_byte_ids` (model.py:48-49).
Implement:
- Byte tokenization: UTF-8 encode, `byte + 1`, truncate to length
- Tensor construction: padding, masking
- The full inference path: context + options → byte tensors → forward pass →
  probabilities → winner + confidence

Test: golden comparison for the full pipeline (context + options in,
probabilities out) against the reference Python model.

### Phase 3: Registration, weight loading, config

- `REGISTER_VLLM_MODEL` entry for cua-s1-forms
- Safetensors weight loader (all tensors float32, no conversion)
- Config loader for `cua-s1-forms.json` (unwrap the envelope, read `config`)
- Server dispatch: the `/v1/score` handler constructs a `CuaS1Inference` call

Test: load the published checkpoint, run a forward pass, verify probabilities
match the reference.

### Phase 4: Server endpoint, reachability

- `/v1/score` endpoint in `api_server.cpp`
- Server dispatch in `server_main.cpp`
- `GET /v1/models` lists cua-s1-forms (already implemented in the Laya branch)
- The endpoint is reachable from the HTTP server on its default configuration

Test: start the server, send a `/v1/score` request, verify the response.

### Phase 5: GPU (CUDA) build — owed

Verify the model compiles and runs on CUDA. The model is f32 throughout, so no
quantization or mixed-precision concerns. Listed as owed, not a pre-PR gate.

## Risks

1. **nn.TransformerEncoderLayer internals**: PyTorch's `TransformerEncoderLayer`
   with `norm_first=True` applies pre-norm: `h = h + attn(norm1(h))` then
   `h = h + ff(norm2(h))`. The attention uses scaled dot-product with
   `src_key_padding_mask`. The feed-forward uses ReLU (the default activation).
   The C++ port must match this exactly, including the order of operations and
   the mask semantics (`True` means "allowed", not "masked").

2. **Packed QKV**: `nn.MultiheadAttention` packs QKV into `in_proj_weight`
   `(3*dim, dim)`. The C++ port must split this into Q, K, V projections or
   use a single matmul with the packed weight. The split order is Q, K, V
   along dimension 0.

3. **Mask semantics**: `src_key_padding_mask` in PyTorch uses `True` for
   padding (masked out) positions. The model code uses
   `src_key_padding_mask=~safe_context_mask` (model.py:218), so `~mask` inverts
   the boolean. The C++ port must apply the same inversion.

4. **Safe mask**: The model forces `mask[:, 0] = True` (model.py:215,
   model.py:225) to avoid all-padding rows, which would produce NaN in
   softmax. The C++ port must replicate this.

5. **finfo.min masking**: The model uses `torch.finfo(scores.dtype).min` for
   masking (model.py:133, model.py:136), not `-inf`. This avoids NaN from
   softmax over all-masked rows. The C++ port must use `std::numeric_limits
   <float>::lowest()` (not negative infinity) for the same reason.

6. **Checkpoint format**: The sidecar JSON wraps the config in an envelope
   with a `state_signature` (SHA-256). The loader does not need to verify
   the signature (that is a Python-side integrity check), but it must read
   through the envelope to reach the `config` object.

## Tests

1. **Model forward golden** (Phase 1): construct a `TinyTransformerScorer`
   with the published checkpoint, run a forward pass on a fixed input, compare
   logits and probabilities against PyTorch output. Tolerance: `1e-5`.

2. **ByteCollator golden** (Phase 2): tokenize a context + options pair, compare
   the resulting tensors (ids, masks) against the Python `ByteCollator` output.

3. **Full pipeline golden** (Phase 2): context + options in, probabilities out,
   compared against the reference Python model end to end.

4. **Weight loading** (Phase 3): load the published checkpoint, verify all
   tensor names and shapes match the expected state dict.

5. **Server endpoint** (Phase 4): start the server, send a `/v1/score` request,
   verify the response format and values match the reference.

6. **Mutation tests**: for each guarantee, mutate the implementation (e.g., skip
   the safe mask, use `-inf` instead of `finfo.min`, swap Q/K split order) and
   verify the test fails.

## Gates

- **Correctness**: logits and probabilities match the reference Python model to
  `1e-5` on the published checkpoint.
- **Build**: CPU build passes with all tests green.
- **Reachability**: `/v1/score` is reachable from the HTTP server on its
  default configuration.
- **GPU (owed)**: CUDA build compiles and runs. Not a pre-PR gate.

## Stop conditions

- The model is correct on CPU (logits and probabilities match the reference).
- All tests pass.
- The `/v1/score` endpoint is reachable from the HTTP server.
- The spec, records, and code are committed and pushed.

## Owed

- GGUF k-quant arms for the encoder (the model is 706K params; quantization is
  not meaningful at this scale, but the standing requirement applies).
- GPU (CUDA) build verification (Phase 5).
- LocalAI backend integration (separate PR in a separate repo).

## Weights

- `cua-ai/cua-s1-forms` @ HuggingFace, revision `f54adbf447f4ca6ec259f529ee3f2e3e09f8cc71`
  - `cua-s1-forms.safetensors` — 2.8 MB, float32 state dict
  - `cua-s1-forms.json` — config + metadata envelope
  - License: MIT

## Git integration

One pull request for spec + implementation, per developer preference.
