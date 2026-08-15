# Speculative-decoding method inventory (vLLM `555967922` / 0.26.0.dev0)

Enumerated from SOURCE, not memory, on 2026-08-09 (`row/SPEC-DECODE-INVENTORY`).
Authoritative surfaces at the parity pin:

- `vllm/config/speculative.py:37-78` — the `SpeculativeMethod` Literal (every
  accepted `method` string) plus `RejectionSampleMethod` / `DraftSampleMethod`.
- `vllm/v1/worker/gpu_model_runner.py:596-655` — the DEFAULT V1 runner's
  method -> proposer dispatch (raises `ValueError` on an unhandled method).
- `vllm/v1/worker/gpu/spec_decode/__init__.py:8-40` — the OPT-IN "Model Runner
  V2" (`VLLM_USE_V2_MODEL_RUNNER=1`) factory; a NARROWER subset (dflash, dspark,
  gemma4_mtp, mtp, eagle/eagle3 only).
- `vllm/v1/spec_decode/` — the V1 proposer classes.
- `vllm/model_executor/models/` — the per-family speculator model classes.
- `vllm/transformers_utils/configs/speculators/algos.py:4-15` — the HF
  `speculators`-library checkpoint-format integration.

**V1/V0 note.** At the pin the engine is V1-only (no V0 model runner). "V1
status" below therefore means: does the DEFAULT V1 runner instantiate a proposer
for the method. `mlp_speculator` is the one method the enum still accepts that
has NO V1 proposer -> it is effectively deprecated / unreachable at the pin.

**Not "ours-beyond-vLLM".** `dflash` (block-diffusion) and `dspark` are REAL
upstream vLLM methods (`vllm/v1/spec_decode/dflash.py`,
`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`); our DFlash port mirrors
vLLM, it is not an addition vLLM lacks.

## Method x status

| Method (`method=`) | Upstream anchor | Draft-weight requirement | Upstream V1 status | Our status (row) |
|---|---|---|---|---|
| `ngram` | `vllm/v1/spec_decode/ngram_proposer.py:12` | none (suffix match over prompt+output) | V1 default runner (not MRV2) | **DONE** `SPEC-NGRAM` (27B 5/5 strict) |
| `ngram_gpu` | `vllm/v1/spec_decode/ngram_proposer_gpu.py:217` | none (on-device n-gram) | V1 default runner | **ABSENT** `SPEC-NGRAM-GPU` |
| `suffix` | `vllm/v1/spec_decode/suffix_decoding.py:9` (Arctic Inference, ext. `arctic_inference`, lazy import :26) | none (suffix tree over prompt+output) | V1 default runner | **ABSENT** `SPEC-SUFFIX` |
| `medusa` | `vllm/v1/spec_decode/medusa.py:18`; model `vllm/model_executor/models/medusa.py` | head-on-base (extra LM heads on target, single non-AR pass) | V1 default runner | **SPIKE** `SPEC-MEDUSA` (W0 only; model row `MODEL-SPEC-medusa-medusa`) |
| `mlp_speculator` | model `vllm/model_executor/models/mlp_speculator.py`; enum `speculative.py:69,890` | separate MLP head | **NONE** — no V1 proposer; dispatch `ValueError` (`gpu_model_runner.py:651-654`); V0-only/deprecated | **ABSENT** `SPEC-MLP-SPECULATOR` (upstream-deprecated) |
| `draft_model` | `vllm/v1/spec_decode/draft_model.py:19` | separate standalone LM (no shared emb/head) | V1 default runner | **ACTIVE** `SPEC-DRAFT-MODEL` (W0+W1 CPU brick; W3 GPU DGX residual) |
| `custom_class` | `vllm/v1/spec_decode/custom_class_proposer.py:12` (class path from `speculative_config.model`) | proposer-defined (pluggable) | V1 default runner | **ABSENT** `SPEC-CUSTOM-CLASS` |
| `eagle` | `vllm/v1/spec_decode/eagle.py:10` (`use_eagle()` `speculative.py:1324-1328`); models `llama_eagle.py`, `cohere_eagle.py`, `mistral_eagle.py`, ... | separate draft + target aux hidden-state tap | V1 default + MRV2 | **ABSENT** `SPEC-EAGLE` (EAGLE1 distinct from eagle3) |
| `eagle3` | same `EagleProposer`; models `llama_eagle3.py`, `qwen3_eagle3.py`, `deepseek_eagle3.py` | separate draft + multi-layer aux taps | V1 default + MRV2 | **BLOCKED** `SPEC-EAGLE3` (no ungated gate-model checkpoint at pin) |
| `extract_hidden_states` | `vllm/v1/spec_decode/extract_hidden_states.py:29` | none (offline hidden-state capture utility) | V1 default runner | **ABSENT** `SPEC-EXTRACT-HIDDEN` |
| `mtp` (canonical) | `EagleProposer` via `use_eagle()`; special `Gemma4Proposer`/`Step3p5MTPProposer` (`gpu_model_runner.py:627-630`); MRV2 `MTPSpeculator` | head-on-base per family (nextn layer on target) | V1 default + MRV2 | **DONE** `SPEC-MTP` (Qwen3.5/3.6 k=1); family breadth `SPEC-MTP-FAMILY` |
| `deepseek_mtp` ... `inkling_mtp` (20 family strings) | `MTPModelTypes` `speculative.py:37-59` -> deprecate-remap to `mtp` (`:686-690`); models `deepseek_mtp.py`, `glm4_moe_mtp.py`, `ernie_mtp.py`, `nemotron_h_mtp.py`, ... | head-on-base per family | V1 default (dispatched by `draft_model_config.hf_config.model_type`) | **ACTIVE** DeepSeek-V4 (`MODEL-SPEC-deepseek-v4-deep-seek-v4-mtp`, W1, weight-blocked); rest **INVENTORIED** (`SPEC-MTP-FAMILY`, model-matrix `MODEL-SPEC-*-mtp`) |
| `dflash` | `vllm/v1/spec_decode/dflash.py` (DFlashProposer); models `laguna_dflash.py`, `qwen3_dflash.py`; speculators algo `algos.py:93` | separate block-diffusion draft + aux | V1 default + MRV2 | **DONE** `SPEC-DFLASH` (ported from vLLM; +GGUF `SPEC-DFLASH-GGUF`) |
| `dspark` | MRV2 `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:37`; models `gemma4_dspark.py`, `qwen3_dspark.py`; speculators algo `algos.py:133` | separate SAR block draft + aux | V1 default (via `use_eagle()`) + MRV2 | ~~**INVENTORIED**~~ **`ACTIVE`** `SPEC-DSPARK` (W1–W8 implemented + GPU-gated; 35B-A3B MoE 0.975x code / 1.012x prose vs the pinned graphed oracle, #442; superseded here 2026-08-12, #536) |

## User-visible config axes (apply on top of a method)

| Axis | Upstream anchor | Values | Our status (row) |
|---|---|---|---|
| Acceptance / rejection sampler | `speculative.py:77,216` (`rejection_sample_method`) | `standard` \| `synthetic` \| `block` | `standard` **ACTIVE** `SPEC-REJECTION`; `synthetic`/`block` **ABSENT** `SPEC-ACCEPT-VARIANTS` |
| Draft sampling | `speculative.py:78,283` (`draft_sample_method`) | `greedy` \| `probabilistic` | `greedy` only; `probabilistic` **ABSENT** `SPEC-ACCEPT-VARIANTS` |
| Dynamic k (per batch size) | `speculative.py:1336-1337`; `vllm/v1/spec_decode/dynamic/utils.py` (`num_speculative_tokens_per_batch_size`) | schedule list | **ABSENT** `SPEC-DYNAMIC` |
| Heterogeneous draft/target vocab (TLI) | `vllm/config/speculative.py:150` (`use_heterogeneous_vocab`); `vllm/v1/spec_decode/vocab_mapping.py:68` (`VocabMapping`); consumed ONLY by `llm_base_proposer.py:432-495,688-691,831-837` (`SpecDecodeBaseProposer`) and `draft_model.py:19,34-58` | cross-tokenizer-family string intersection, target↔draft ID translation, constrained draft logits | **INVENTORIED** `SPEC-TLI` — untouched, and it hangs off the V1 `SpecDecodeBaseProposer`, NOT the V2-runner DFlash/DSpark speculators we ported, so its host row is `SPEC-DRAFT-MODEL` and it is prerequisite-blocked behind that row's W3 (2026-08-12, #536). DSpark's `d2t` does NOT cover it: `d2t` offsets ids inside ONE vocabulary |
| GDN spec metadata + slot-snapshot rollback | `vllm/v1/attention/backends/gdn_attn.py` | — | **ACTIVE** `SPEC-GDN-SEGMENTS` |

## HF `speculators` checkpoint-format integration

`vllm/transformers_utils/configs/speculators/algos.py` registers
`SUPPORTED_SPECULATORS_TYPES = {eagle3, peagle, dflash, dspark}` (`:15,55,93,133`)
and rewrites a HF `speculators`-format config into a vLLM draft arch. ~~Our side
loads native draft configs directly (no `speculators`-format adapter); tracked
under the EAGLE/DFlash/DSpark rows as a loader-format residual, not a separate
method.~~ **SUPERSEDED 2026-08-12 (#536): `SPEC-DSPARK` W3 landed the DSpark
adapter** — `speculators_model_type == "dspark"` detection, the
`speculators_config` proposal-method unwrap, and the `algos.py:133-165` field
translation (`aux_hidden_state_layer_ids` → `target_layer_ids = [i-1]`,
`sample_from_anchor`, `draft_vocab_size`, `mask_token_id`, `markov_rank`,
`block_size`), in `src/vllm/model_executor/models/qwen3_dspark.cpp:227-300`.
It is the path the `RedHatAI/*speculator.dspark` gate-model drafts load through.
The residual is now the EAGLE3/PEAGLE/DFlash algos, not the whole subsystem.

## Our lifecycle summary (engine matrix §Speculative decoding)

- **DONE**: `SPEC-MTP`, `SPEC-MTP-GGUF`, `SPEC-DFLASH`, `SPEC-DFLASH-GGUF`.
- **ACTIVE**: `SPEC-REJECTION`, `SPEC-GDN-SEGMENTS`, `SPEC-NGRAM`,
  `SPEC-DRAFT-MODEL`, **`SPEC-DSPARK`** (moved out of INVENTORIED here
  2026-08-12, #536: it has been `ACTIVE` in the engine matrix since the
  2026-08-09 spike, with W1–W8 implemented and GPU-gated).
- **SPIKE**: `SPEC-MEDUSA`. **BLOCKED**: `SPEC-EAGLE3`.
- **INVENTORIED**: ~~`SPEC-DSPARK`,~~ `SPEC-TLI`, and the nine enumerated here:
  `SPEC-NGRAM-GPU`, `SPEC-SUFFIX`, `SPEC-EAGLE`, `SPEC-MTP-FAMILY`,
  `SPEC-ACCEPT-VARIANTS`, `SPEC-DYNAMIC`, `SPEC-CUSTOM-CLASS`,
  `SPEC-EXTRACT-HIDDEN`, `SPEC-MLP-SPECULATOR`.

Each INVENTORIED row is spike-first: a `.agents/specs/` spike gate precedes any
implementation claim, per the tabular-inventory directive. Value order for the
nine gaps (gate-model-usable and draft-free first): `SPEC-NGRAM-GPU` ->
`SPEC-SUFFIX` -> `SPEC-EAGLE` -> `SPEC-MTP-FAMILY` -> `SPEC-ACCEPT-VARIANTS` ->
`SPEC-DYNAMIC` -> `SPEC-CUSTOM-CLASS` -> `SPEC-EXTRACT-HIDDEN` ->
`SPEC-MLP-SPECULATOR` (upstream-deprecated, completeness only).
