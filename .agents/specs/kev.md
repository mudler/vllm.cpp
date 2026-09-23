# SPEC — `MODEL-KEV`: kev System 1 decision model (Qwen3.5 + LoRA + PointerHead)

Port `jaredpalmer/kev-0.8b` into vllm.cpp as the third SystemOne-class model,
reusing the `/v1/systemone` API (choice/score/noul question types) already on
`main` from the GLiNER2.5 work. kev is a frozen Qwen3.5-0.8B-Base backbone
with a rank-16 LoRA adapter merged at load time and a PointerHead readout. CPU +
GPU (CUDA), OpenAI-compatible serving.

## Now

`ACTIVE` — Phases 1-4 committed (sequence construction, readout, LoRA merge,
PointerHead, golden-vector tests: 25 cases, 126 assertions). Phase 5a
(`ForwardDenseHidden`) and Phase 5b (model registration, `convert-kev.py`,
`REGISTER_VLLM_MODEL`) committed. Phase 5c inference pipeline (`KevInference`
in `kev_registry.cpp`) defined and compiling but not yet wired to
`/v1/systemone` — the server dispatch depends on the `DecisionFn` callback
from Laya PR #3263, which has not merged. GPU build verification is owed.

## Scope

- **Row.** `MODEL-KEV` (this spec). New model-matrix row under `MODEL-TOKCLS`
  (kev answers structured questions, same SystemOne class as GLiNER2.5 and Laya).
- **In.** Qwen3.5-0.8B-Base backbone (24 layers: 18 DeltaNet + 6 full-attention,
  hidden=1024) in feature-extraction mode (ForwardHidden, no LM head); rank-16
  LoRA merge at load time (372 F32 tensors, scaling=2.0); PointerHead readout
  (q/k Linear 1024 to 256, scale=1/sqrt(256), temperature=2.406); sequence
  construction with Qwen special-token delimiters; model registration via
  `REGISTER_VLLM_MODEL`; `/v1/systemone` dispatch for kev; config loading from
  `adapter_config.json` + sidecar `head.safetensors`; CPU build + tests; GPU
  forward (CUDA).
- **Out (owned by other rows).** ROCm kernel tuning. GGUF k-quants (owed, named
  below). Runtime prompt-activated LoRA (kev merges at load, not at runtime).
  Prefix caching optimization (deferred — kev reprocesses state per question).
  LocalAI backend is a separate PR in a separate repo.
- **Reuse.** The Qwen3.5 dense backbone forward path (`DenseForwardLayers`,
  `qwen3_5.cpp`). The `/v1/systemone` API endpoints, request/response structs,
  and server dispatch from GLiNER2.5. The `DecisionFn` callback mechanism from
  Laya. The Qwen2 BPE tokenizer already supported in the tree.

## Upstream anchors

### Oracle: kev reference implementation

Repository: `jaredpalmer/kev` @ `19dcae9b6e3e1a48200c5825aad9fc200d31e20a`.

- `kev/model.py`: `SPECIAL` (token IDs), `encode()` (sequence construction),
  `PointerHead`, `DecisionModel`, `forward` / `forward_rows_batch`.
- `kev/api.py`: `Noul`, `Choice`, `Score`, `Question`, `SystemOneRequest`,
  `to_answers`. Confidence formulas (DIFFER from Laya).
- `kev/checkpoint.py`: `Meta`, `Checkpoint.load` (merges LoRA into base, loads
  head).
- `kev/serve.py`: `Server`, systemone endpoint, prefix-cache LRU.

### Base model: Qwen3.5-0.8B-Base

- `Qwen/Qwen3.5-0.8B-Base` @ `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- hidden_size=1024, 24 layers, 8 attn heads, 2 KV heads, head_dim=256.
- intermediate_size=3584, vocab=248320, silu, rms_norm_eps=1e-6.
- tie_word_embeddings=true.
- layer_types: 18 `linear_attention` (DeltaNet) + 6 `full_attention` (idx
  3,7,11,15,19,23).
- RoPE: theta 1e7, mrope_interleaved=true, section [11,11,10],
  partial_rotary_factor=0.25.
- Already fully implemented in vllm.cpp: `qwen3_5.cpp`, `qwen3_5_dense.cpp`,
  `qwen3_5_weights.cpp`.

### Adapter: jaredpalmer/kev-0.8b

- `adapter_config.json` — peft LoRA config: r=16, alpha=32, scaling=2.0.
- `adapter_model.safetensors` — 43.3 MB, 372 F32 LoRA tensors.
- `head.pt` — 2.1 MB torch pickle (PointerHead weights + temperature).
  Conversion script produces `head.safetensors` + `meta.json`.
- `tokenizer.json` — standard Qwen2 BPE (already supported).

## Design

### Phase 1: head.pt conversion + PointerHead host forward + golden tests

- Script: `scripts/convert-kev-head.py` — loads `head.pt` via torch, saves
  `head.safetensors` (4 tensors: q.weight [256,1024], q.bias [256],
  k.weight [256,1024], k.bias [256]) + `meta.json` (temperature=2.406,
  head_dim=256, base model revision).
- PointerHead forward: `logits[opt] = (k(h_option) . q(h_decide)) * scale /
  temperature`; `softmax(logits)` over options gives the output distribution.
  `scale = 1/sqrt(256)`.
- Golden tests: generate from reference implementation with known hidden states.
  Tests live in `tests/vllm/models/test_kev.cpp`.

### Phase 2: LoRA merge at load time

- Load base Qwen3.5-0.8B-Base weights via existing `LoadQwen3_5Dense`.
- Load 372 F32 LoRA tensors from `adapter_model.safetensors`.
- Merge: `W' = W + 2.0 * (lora_B @ lora_A)` for each target tensor.
- LoRA naming: `base_model.model.layers.{i}.<group>.<proj>.lora_A.weight` /
  `lora_B.weight`. Base naming: `model.language_model.layers.{i}.<group>.<proj>.weight`.
- Target modules: self_attn (q/k/v/o_proj on 6 full-attn layers), linear_attn
  (in_proj_qkv/z/a/b, out_proj on 18 DeltaNet layers), mlp (gate/up/down_proj
  on all 24 layers).
- This is a one-time operation at load, NOT runtime LoRA serving. Merge in fp32,
  store result in the base weight dtype (bf16).

### Phase 3: ForwardHidden for Qwen3.5 Dense

- Add `ForwardHidden` to `Qwen3_5DenseModel` (declare in `qwen3_5_dense.h`,
  implement in `qwen3_5.cpp`).
- Mirrors Qwen3's `ForwardHidden` (`qwen3.cpp:570-592`): calls
  `DenseForwardLayers` with a return-hidden flag, skips `lm_head`.
- Returns `[n_out, hidden_size]` f32 rows (post-final-RMSNorm hidden states).
- Precedent: `LlamaEmbeddingLoadedModel` uses `ForwardHidden` for pooling.

### Phase 4: Sequence construction + inference pipeline

- `encode()`: construct `[state_tokens + question_tokens]` using Qwen special
  tokens as delimiters:
  - `<|fim_prefix|>` (state-start), `<|fim_middle|>` (question),
    `<|box_start|>` (option-start), `<|box_end|>` (option-end), `<|decide|>`.
  - Layout: `[<state> state_tokens...]` then per question:
    `[<q> instr <opt> o1 </opt> <opt> o2 </opt> ... <decide>]`.
  - Track `decide_idx[k]` and `opt_idx[k][j]` positions.
- **Simplest correct approach** (no prefix caching): for each question,
  construct `[state + question_tokens]` and run the full sequence through
  `ForwardHidden`. Extract hidden states at decide and opt positions. Apply
  PointerHead.
- DeltaNet layers are recurrent: running `[state + question]` as a single
  causal sequence naturally gives question tokens access to state tokens.
  Isolation between questions is by construction (separate forward passes).
- Prefix caching optimization deferred to future work.

### Phase 5: Registration, server dispatch, /v1/systemone endpoint

- Register via `REGISTER_VLLM_MODEL` (mirror `llama_embedding_registry.cpp`).
- `LoadedModel` subclass owning merged Qwen3.5 weights + PointerHead weights.
- `is_pooling_model=true`, `is_text_generation_model=false`.
- `/v1/systemone` dispatch: reuse the `DecisionFn` callback from Laya (PR #3263).
  kev plugs in as an alternative model behind the same API.
- Confidence formulas DIFFER from Laya:
  - choice: `(max(p) - 1/K) / (1 - 1/K)` (already on main as
    `ChoiceConfidence`)
  - score: `1 - E|level - mode| / (L - 1)` (already on main as
    `ScoreConfidence`)
  - noul: just `p(true)`, no confidence
- All probabilities rounded to 2 decimals (same as Laya).

### Phase 6: E2E parity test vs reference

- Run `kev.serve` Python server on identical inputs (choice/score/noul).
- Compare outputs. Token-exact where possible; near-tie robust for bf16 paths.
- Server E2E test through `/v1/systemone` HTTP endpoint.

## Risks

- DeltaNet layers in Qwen3.5: the existing forward path may not expose the
  recurrent state needed for prefix caching. The simple approach (full
  reprocessing per question) avoids this.
- `head.pt` is torch pickle: conversion script needs torch. This is a
  build-time preprocessing step, not a runtime dependency.
- LoRA merge changes weight tensor values: verify merge is correct by comparing
  merged weights against reference.
- The Qwen3.5-0.8B-Base checkpoint (~1.6 GB bf16) must be available.
- The `DecisionFn` callback mechanism is in the Laya PR (#3263). kev's Phase 5
  server dispatch depends on Laya merging first. Phases 1-4 have no such
  dependency.

## Gates

- CPU-correct: all golden tests pass (PointerHead, LoRA merge, ForwardHidden,
  end-to-end).
- E2E parity: choice/score/noul outputs match reference within tolerance.
- Reachability: `/v1/systemone` endpoint serves kev model through
  `ModelRegistry::Forward`.
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
- Prefix caching optimization.
- Server dispatch via `DecisionFn` callback (blocked on Laya PR #3263):
  `KevInference` is defined in `kev_registry.cpp` and compiles, but is not
  yet called from `/v1/systemone`. The device queue is stored on
  `KevLoadedModel` during `PrepareKev` (the `ModelFactory::prepare`
  callback, which already receives `vt::Queue&`) and retrieved by
  `KevInference` from the model — no shared-header accessor needed.
  Only the server dispatch code in `server_main.cpp` remains, blocked on
  Laya PR #3263 landing the `DecisionFn` callback and `set_decision` API.

## Git integration

One pull request (repository default policy). Spec commit precedes
implementation commits in the same pull request.

## Weights

- Base: `Qwen/Qwen3.5-0.8B-Base` @ `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
  — bf16, ~1.67 GB.
- Adapter: `jaredpalmer/kev-0.8b` — `adapter_model.safetensors` (43.3 MB, F32
  LoRA), `head.pt` (2.1 MB, converted to `head.safetensors`), `tokenizer.json`.
