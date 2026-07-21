# SPIKE: Breadth-Sweep Model Bring-Up — OPT-125m (`OPTForCausalLM`)

*Breadth-sweep Tier-1 rank 4 (`.agents/specs/breadth-sweep-plan.md` §B.3), the
CROSS-FAMILY additivity canary. Row `MODEL-TEXT-opt-optfor-causal-lm`
(model-matrix, `INVENTORIED` → `ACTIVE`). Template:
`.agents/specs/sweep-qwen3-coder-30b.md`.*
*Grounding pins: vLLM `/home/mudler/_git/vllm` @ `e24d1b24`; our tree @ `c56ab28`.
Standards: PR-#4 additivity ([[extensibility-first-additive-hw-models]]),
gate selection by measurement ([[near-tie-distributional-gate]]), mirror-vLLM.*

---

## 0. Purpose and headline

Every model in this tree is a Qwen variant. OPT is deliberately chosen as the
HONEST test of the claim that *"a new model family = new files"*, because it
shares almost nothing with them: no RoPE (learned absolute positions with a
fairseq offset of 2), biased projections, LayerNorm instead of RMSNorm, a plain
ReLU fc1/fc2 MLP instead of SwiGLU, a `do_layer_norm_before` pre/post-LN switch,
and tied embeddings by default.

**Headline: the MODEL seams held completely; two layers UNDERNEATH them leaked.**

- The whole model layer — registry, config, loader, forward, KV-cache spec —
  landed as **4 new files with ZERO edit to any shared model-layer file**. No
  runner change, no scheduler change, no platform change, no attention-registry
  change, no CUDA kernel change, and no edit to any existing model.
- What leaked is one level down, in two places that had been *implicitly*
  specialized to Qwen without anyone noticing: the **`vt::` op table** (no
  LayerNorm, no ReLU, no elementwise/bias Add) and the **tokenizer** (no GPT-2
  byte-level split; a parsed-but-never-applied post-processor BOS). Both leaks
  were absorbed as APPEND-ONLY extensions of an existing extension point, and
  both are now generic for the next family.
- The tokenizer BOS leak was **correctness-fatal and silent**: without it the
  gate scored 0/6 prompts with fluent-looking output. It was caught only because
  vLLM's own `prompt_token_ids` are committed as goldens alongside the
  continuations. That cross-check is the reusable methodology lesson.

**Result: STRICT token-exact 6/6 prompts, 96/96 tokens vs the vLLM 0.25.0
oracle**, on a model where vLLM's own greedy was MEASURED deterministic.

---

## 1. TARGET + checkpoint reality (verified on dgx 2026-07-21)

`.agents/specs/breadth-sweep-plan.md` §B.1 lists `facebook--opt-125m` as
"present". **That premise was wrong and is corrected here:** the HF cache held
only a `config.json` symlink (36 KB total, no weights). The rest was downloaded,
and what arrived is not directly loadable by us either — see D2.

`config.json` (facebook/opt-125m):

| field | value | consequence |
|---|---|---|
| `architectures` | **`OPTForCausalLM`** | maps to `opt` (registry.py:176) |
| `hidden_size` / `num_hidden_layers` | 768 / 12 | tiny — cheap oracle + gate |
| `num_attention_heads` | 12 | **head_dim 64**, and NO `num_key_value_heads` (OPT predates GQA) |
| `ffn_dim` | **3072** | NOT `intermediate_size` — our typed HfConfig leaves that 0 |
| `activation_function` | **`relu`** | not SwiGLU |
| `do_layer_norm_before` | **true** | PRE-LN placement (the 125m/1.7B/…/175B branch) |
| `word_embed_proj_dim` | 768 (== hidden) | no project_in/project_out |
| `max_position_embeddings` | 2048 | LEARNED table is **[2050, 768]** (offset 2) |
| `torch_dtype` | **`float16`** | our CUDA compute path is bf16/f32 — see D1 |
| rope fields | **absent** | HfConfig still synthesizes them — see L3 |
| `enable_bias`, `layer_norm_elementwise_affine`, `tie_word_embeddings`, `_remove_final_layer_norm` | absent | fall back to transformers `OPTConfig` defaults: true, true, true, false |

Checkpoint tensors: **197** = 5 top-level (`embed_tokens`, `embed_positions`,
`final_layer_norm.{weight,bias}`, plus the tied-and-skipped `lm_head.weight`) +
12 layers × 16 (q/k/v/out weight+bias = 8, fc1/fc2 weight+bias = 4, two
LayerNorms weight+bias = 4).

---

## 2. Upstream grounding — `vllm/model_executor/models/opt.py @ e24d1b24`

- **`OPTLearnedPositionalEmbedding` (:59-68):** `nn.Embedding(num_embeddings +
  2, dim)`; `forward(positions)` returns `super().forward(positions + 2)`. The
  offset is the fairseq padding-idx hack, and it is why the on-disk table is
  `[max_position_embeddings + 2, H]`.
- **`OPTAttention` (:71-122):** `QKVParallelLinear(..., bias=config.enable_bias)`
  chunked into q,k,v; `RowParallelLinear` out_proj with the same bias;
  `scaling = head_dim ** -0.5`. **No q/k norm, no RoPE.**
- **`OPTDecoderLayer` (:125-195):** two `nn.LayerNorm(embed_dim,
  elementwise_affine=...)`; `fc1`/`fc2` Column/Row parallel with bias;
  `get_act_fn(config.activation_function)`. `forward` (:168-195) branches on
  `do_layer_norm_before` for BOTH the attention and the FC sub-blocks.
- **`OPTDecoder` (:198-293):** embed_tokens + embed_positions summed (:274-279);
  project_in/project_out only when `word_embed_proj_dim != hidden_size`
  (:220-241); decoder `final_layer_norm` only when `do_layer_norm_before and not
  _remove_final_layer_norm` (:243-253).
- **`OPTForCausalLM` (:327-394):** `hf_to_vllm_mapper` renames `decoder.` →
  `model.decoder.` and stacks q/k/v (:328-338); `packed_modules_mapping`
  (:339-341); `lm_head = embed_tokens` when `tie_word_embeddings` (:352-353)
  with `skip_prefixes=["lm_head.weight"]` in the loader (:390-392).

---

## 3. ADDITIVITY LEDGER (the primary deliverable)

### (a) NEW FILES the bring-up ADDS

| New file | Purpose |
|---|---|
| `include/vllm/model_executor/models/opt.h` | `OPTConfigExtras` + weight PODs + `OPTModel` decls |
| `src/vllm/model_executor/models/opt_registry.cpp` | `REGISTER_VLLM_MODEL(opt, "OPTForCausalLM")`, config hook, KV spec, factory |
| `src/vllm/model_executor/models/opt_weights.cpp` | BF16 safetensors loader (merged qkv + merged bias, LayerNorm weight+bias, offset-2 position table) |
| `src/vllm/model_executor/models/opt.cpp` | the forward |
| `src/vt/cuda/cuda_layernorm.cu` | CUDA LayerNorm / ReLU / Add (self-registering TU) |
| `src/vt/cpu/cpu_layernorm.cpp` | CPU LayerNorm / ReLU / Add (self-registering TU) |
| `tests/vt/test_ops_layernorm.cpp` | new-op unit tests |
| `tests/vllm/models/test_opt_load.cpp` | W2 loader gate |
| `tests/vllm/models/test_opt_paged_engine.cpp` | W4 SACRED correctness gate |
| `scripts/opt-materialize-checkpoint.py` | pickle+fp16 → safetensors+bf16 + fast tokenizer |
| `scripts/opt-oracle-capture.py` | oracle capture + self-determinism report |
| `tests/parity/goldens/opt_greedy/` | `greedy_ids.npy`, `greedy_dist.npy`, `p{i}_prompt.i32` |

### (b) SHARED FILES it must TOUCH — complete and classified

| # | Shared file | Edit | Class |
|---|---|---|---|
| 1 | `include/vt/ops.h` | +3 `OpId` values, +3 `Fn` typedefs, +`LayerNormArgs`, +3 decls | **APPEND-ONLY extension point** (L2) |
| 2 | `src/vt/ops.cpp` | +3 validating dispatch wrappers | **APPEND-ONLY extension point** (L2) |
| 3 | `CMakeLists.txt` | +5 source lines | build-list mechanics (explicit list, no glob) |
| 4 | `tests/CMakeLists.txt` | +3 test registrations | build-list mechanics |
| 5 | `include/vllm/model_executor/models/dense_weight_loaders.h` | +`LoadMergedBf16Vector` | **APPEND-ONLY**, generic (merged bias vectors) |
| 6 | `include/vllm/tokenizer/pretokenizer.h` | +`SplitPattern::kGpt2` | **LEAK** (L4a) |
| 7 | `src/vllm/tokenizer/pretokenizer.cpp` | +GPT-2 scanner; existing scanner untouched | **LEAK** (L4a), append-only |
| 8 | `src/vllm/tokenizer/tokenizer.cpp` | accept `use_regex=true`; detect `kGpt2`; +`EncodeWithSpecialTokens` | **LEAK** (L4a+L4b) |
| 9 | `include/vllm/tokenizer/tokenizer.h` | +1 method decl | **LEAK** (L4b) |
| 10 | `src/vllm/v1/engine/input_processor.cpp` | **1 line**: `Encode` → `EncodeWithSpecialTokens` | **LEAK** (L4b), behavior-preserving for Qwen |
| 11 | `tests/vllm/models/test_model_registry.cpp` | arch count 4→5, sort order, +2 cases | expected test-surface growth |

**ZERO edits to:** the runner (`v1/worker/gpu/runner.cpp`), the scheduler, the
platforms, the attention-backend registry, `hf_config.{h,cpp}`, **any CUDA
kernel**, `dense_attn_block.h` (reused, not modified), and every existing model
file (`qwen3*.{h,cpp}`, `qwen3_5*.{h,cpp}`).

### (c) SEAMS THAT HELD

- **S1 — model registry (`REGISTER_VLLM_MODEL`).** Flawless. One new TU, one
  macro line, zero shared-array edit; the registry went 4→5 architectures with
  only count/order assertions changing. This is the seam the sweep was betting
  on and it paid off for a family that shares nothing with the incumbents.
- **S2 — `HfConfig::raw` escape hatch.** All eight OPT-only config fields
  (`ffn_dim`, `word_embed_proj_dim`, `do_layer_norm_before`, `enable_bias`,
  `layer_norm_elementwise_affine`, `_remove_final_layer_norm`,
  `tie_word_embeddings`, `activation_function`) are read from `raw` by
  `GetOPTConfigExtras`, so the shared typed `HfConfig` POD was **not widened at
  all** — even though OPT does not even use `intermediate_size`.
- **S3 — model-shape-agnostic runner + full-attention-only KV spec.** Zero
  runner change, exactly as the Qwen3-dense bring-up predicted. `is_dense_model`
  + one `FullAttentionSpec` group was the entire integration.
- **S4 — `dense_weight_loaders.h`.** `LoadBf16Direct` / `LoadBf16Transposed` /
  `LoadMergedBf16RawNK` were reused verbatim across a family boundary; only one
  genuinely new helper was needed (merged rank-1 bias vectors), added
  append-only.
- **S5 — the device glue in `dense_attn_block.h`.** `Dev`/`DBuf`/the shared
  `DevicePool`/`ResidentWeight`/`KvSlice`/`MakeTensor`/`Reshape` were reused
  verbatim by a non-Qwen forward. This is the part of that header that actually
  generalized.
- **S6 — paged attention at a new head_dim.** OPT's `head_dim == 64` runs
  through the generic `LaunchPaged` path with **no kernel edit**: the FA2 fast
  paths are opt-in gates on `d == 128` / `d == 256`, so a new head size degrades
  to correct-but-unoptimized rather than failing. Correctness is additive;
  SPEED is the open increment (§7).

### (d) SEAMS THAT LEAKED

- **L1 — `dense_attn::AttnBlock` does NOT stretch across families.** It
  hard-codes the Qwen attention preamble: per-head q/k RMSNorm, NeoX RoPE, and
  an explicit `VT_CHECK(w.qkv_bias.Empty())`. OPT has none of the first two and
  REQUIRES the third to be non-empty. **This was NOT patched around** — OPT has
  its own `OPTAttnBlock` and reuses only the glue underneath (S5). Honest
  reading: what we called "the shared dense attention block" is really *the
  shared Qwen attention block*; the reusable unit is one layer lower than the
  name suggests.
- **L2 — the `vt::` op vocabulary was Qwen-shaped.** No LayerNorm (only
  RMSNorm), no ReLU (only SiLU/SwiGLU), and no elementwise or bias-broadcast Add
  (the Qwen path folds residuals into `fused_add_rms_norm` and has bias-free
  projections). Three new ops were needed. The op-table pattern absorbed them
  append-only and the kernels live in two NEW self-registering TUs — but the
  enum, the typedefs and the dispatch wrappers are in two SHARED files, so **a
  new family cannot be purely new files while the op table is centralized.**
  This is a design property, not a defect; it is recorded so the claim stays
  honest.
- **L3 — `HfConfig` presumes RoPE.** For a model with no rotary embedding at
  all, `LoadHfConfig` still synthesizes `rotary_dim = partial_rotary_factor *
  head_dim = 64` and `rope_theta = 10000`. Harmless here only because the OPT
  forward never reads them. Pinned by an explicit assertion in
  `test_opt_load.cpp` so the trap is documented rather than latent for the next
  no-RoPE family.
- **L4 — the tokenizer had TWO cross-family gaps, both correctness-fatal.**
  - **L4a — no GPT-2 byte-level split.** OPT's `tokenizer.json` carries
    `ByteLevel{use_regex: true}` with NO explicit `Split` component; we
    explicitly `Fail`ed on it. The original GPT-2 regex differs from the
    Qwen/Llama-3 one in **four of six alternatives** (case-SENSITIVE
    contractions; a plain ` ?` prefix rather than `[^\r\n\p{L}\p{N}]?`;
    UNBOUNDED `\p{N}+` digit runs; no `[\r\n]*` punct tail and no
    `\s*[\r\n]+` rule), so it got its own scanner rather than more flags.
  - **L4b — the post-processor BOS was parsed but never applied.** `BosId()`
    correctly returned 2 from OPT's `TemplateProcessing`, but `Encode()` is
    documented "No BOS/EOS added (caller policy)" and **no caller had a
    policy** — because no Qwen tokenizer declares one. Fixed with
    `EncodeWithSpecialTokens` (HF's `add_special_tokens=True`) called from
    `input_processor.cpp`; byte-identical for every Qwen model, whose bos/eos
    are both -1.
  - **Why this matters more than the code:** without L4b the gate scored
    **0/6 prompts** while emitting fluent, plausible English. The bug was
    localized in one run only because vLLM's own `prompt_token_ids` are
    committed as goldens and compared per prompt. **Committing the oracle's
    tokenization, not just its continuations, is what turned a silent
    whole-model failure into a one-line diff.**
- **L5 — checkpoint format.** The HF snapshot ships a torch-pickle
  `pytorch_model.bin` (no safetensors) and GPT-2 `vocab.json`/`merges.txt` (no
  `tokenizer.json`). Our loader reads neither. Handled OUT of tree by a
  committed materialization script (D2) rather than by widening the loader — but
  it is a real gap for older-family checkpoints generally.
- **L6 — no fp16 compute path.** OPT ships `torch_dtype: float16`, and `kF16` is
  unimplemented across our CUDA compute path (it appears only in `cuda_gdn.cu`).
  Both arms therefore run bf16 (D1). Recorded as a genuine coverage gap: we
  cannot today run an fp16-native checkpoint in its native dtype.

### (e) Honest count & verdict vs PR-#4

PR #4 (the RTX 5070 sm_120 bring-up) scattered +286 lines into `qwen3_5.cpp`,
+92 into `cuda_sample.cu`, +59 into `runner.cpp` — edits to shared *behavior*.

OPT touches 11 shared files, but the character is categorically different:
**9 of 11 are append-only additions to an existing extension point or build
list, 1 is a test-surface count, and exactly 1 (`input_processor.cpp`) changes a
line of shared behavior** — and that one is provably a no-op for every existing
model (Qwen bos/eos are both -1). **No shared file's existing logic was
modified, and no existing model's numerics moved.**

**Verdict: a genuinely new family lands as ~85% new files. The model-layer
seams (registry, config, loader, runner, KV) held 100%. The leaks are all in
lower shared vocabularies — the op table and the tokenizer — which had been
silently specialized to Qwen, and both are now generic.** The next non-Qwen
family (GPT-2, BLOOM, Falcon, Llama) inherits LayerNorm/ReLU/Add, the merged
bias loader, the GPT-2 pre-tokenizer and the special-token encode path, so its
leak surface should be materially smaller.

---

## 4. Structured contract

### Scope

Add `OPTForCausalLM` (facebook/opt-125m, 12L dense, BF16) end to end — registry,
config parse, safetensors loader, paged forward, correctness gate — as the
cross-family additivity canary for row `MODEL-TEXT-opt-optfor-causal-lm`. In
scope: the PRE-LN and POST-LN `do_layer_norm_before` branches, tied and untied
lm_head, biased and (defensively) unbiased projections, `elementwise_affine`
on/off. OUT of scope: `word_embed_proj_dim != hidden_size`
(project_in/project_out) and non-relu activations — both rejected loudly by
`ParseOPTConfig` because no checkpoint exists on the box to gate them (R2); LoRA
and pipeline parallel (upstream `SupportsLoRA`/`SupportsPP`, outside our consumed
`ModelInfo`); GGUF; fp16-native compute (L6); the SPEED bar (§7).

### Upstream chain

`vllm/model_executor/models/opt.py` @ `e24d1b24` (`OPTLearnedPositionalEmbedding`
:59-68, `OPTAttention` :71-122, `OPTDecoderLayer` :125-195, `OPTDecoder`
:198-293, `OPTModel` :296-324, `OPTForCausalLM` :327-394), registered at
`vllm/model_executor/models/registry.py:176`. Layer primitives:
`layers/activation.py::get_act_fn` (relu), `layers/linear.py::{QKVParallelLinear,
ColumnParallelLinear, RowParallelLinear, ReplicatedLinear}`,
`layers/vocab_parallel_embedding.py::{VocabParallelEmbedding, ParallelLMHead}`,
`models/utils.py::{AutoWeightsLoader, WeightsMapper}`. LayerNorm is torch-native
(`nn.LayerNorm` → ATen `native_layer_norm` /
`aten/src/ATen/native/cuda/layer_norm_kernel.cu`), the semantics our
`vt::LayerNorm` mirrors. Attention/KV run through our already-ported paged
attention; no upstream attention change is involved.

### Our baseline

Landed seams reused as-is: `REGISTER_VLLM_MODEL` self-registration
(`model_registry.h:167-189`), the model-shape-agnostic runner
(`runner.cpp:458-470,651-680`), `dense_weight_loaders.h`, the device glue in
`dense_attn_block.h` (`Dev`/`DBuf`/`DevicePool`/`ResidentWeight`/`KvSlice`), the
generic `LaunchPaged` path (`cuda_paged_attn.cu:2513-2704`), Platform +
attention-backend registries. Precedents: `.agents/specs/
first-additive-model-qwen3-dense.md` (dense template) and
`.agents/specs/sweep-qwen3-coder-30b.md` (the W-series protocol this follows).
NOT reused: `dense_attn::AttnBlock` and `dense_attn::BuildStepInputs` (L1).

### Port map

| Upstream | Ours |
|---|---|
| `registry.py:176` entry | `opt_registry.cpp` `REGISTER_VLLM_MODEL(opt, "OPTForCausalLM")` |
| `OPTConfig` fields + defaults | `GetOPTConfigExtras` / `ParseOPTConfig` (`opt_registry.cpp`) via `HfConfig::raw` |
| `hf_to_vllm_mapper` + `packed_modules_mapping` (:328-341) | `LoadOPTForCausalLMWeights` (`opt_weights.cpp`) — merged qkv weight + merged qkv bias |
| `skip_prefixes=["lm_head.weight"]` (:390-392) | tied branch in `LoadOPTForCausalLMWeights`; forward aliases `embed_tokens` |
| `OPTLearnedPositionalEmbedding` (:59-68) | host-side `positions + 2` in `BuildOPTStepInputs` + `vt::Embedding` over the `[P+2,H]` table |
| `OPTAttention.forward` (:114-122) | `OPTAttnBlock` (`opt.cpp`): `BiasedProj` → `vt::QkvSplit` → `ReshapeAndCache` → `vt::PagedAttention` → `BiasedProj` |
| `nn.LayerNorm` (:146-148,164-166,248-251) | **NEW** `vt::LayerNorm` (`cuda_layernorm.cu` / `cpu_layernorm.cpp`) |
| `get_act_fn("relu")` (:156) | **NEW** `vt::Relu` |
| bias adds + residual joins (:90-104,149-163,178,191,279) | **NEW** `vt::Add` (row-broadcast + elementwise) |
| `OPTDecoderLayer.forward` do_layer_norm_before branch (:168-195) | `RunLayer` (`opt.cpp`), both branches |
| `OPTDecoder.forward` (:266-293) | `ForwardBody` (`opt.cpp`) |
| `ByteLevel(use_regex=true)` GPT-2 split | **NEW** `SplitPattern::kGpt2` + `PretokenizeGpt2` |
| HF `encode(add_special_tokens=True)` | **NEW** `Tokenizer::EncodeWithSpecialTokens`, called from `input_processor.cpp` |

### Tests to port

| Upstream test | Tier | Ours |
|---|---|---|
| `tests/models/language/generation/test_common.py:77` (`facebook/opt-125m` greedy vs reference) | T-parity | `tests/vllm/models/test_opt_paged_engine.cpp` — STRICT token-exact vs the vLLM 0.25.0 oracle |
| `tests/models/registry.py:448` + `tests/models/test_registry.py` (arch resolution) | T-unit | `tests/vllm/models/test_model_registry.cpp` — OPT resolve + KV spec + config-hook cases |
| loader half of the above (name map/shapes) | T-unit | `tests/vllm/models/test_opt_load.cpp` — all 197 tensors mapped, shapes, no leftover |
| `tests/kernels/core/test_activation.py` (get_act_fn) + torch `nn.LayerNorm` semantics | T-unit | `tests/vt/test_ops_layernorm.cpp` — LayerNorm/Relu/Add vs the torch reference, CPU + CUDA |

Nothing is checked in SKIPPED: the two out-of-scope upstream variants
(project_in/out, non-relu) are rejected by `ParseOPTConfig` and that rejection is
itself asserted.

### Gates

1. **Correctness (SACRED).** Gate SELECTED by measurement per
   [[near-tie-distributional-gate]]: `scripts/opt-oracle-capture.py --runs 5`
   found vLLM's own greedy **DETERMINISTIC on all 6 prompts, 0 multi-valued
   (prompt,pos) cells** ⇒ **STRICT token-exact** is the bar (no near-tie band).
   Evidence committed in `greedy_dist.npy` and re-asserted by the test.
2. **Loader.** All 197 tensors mapped with correct shapes, nothing leftover,
   `lm_head.weight` the one deliberate (tied) skip.
3. **New ops.** `test_ops_layernorm` green on CPU and CUDA.
4. **Regression, non-negotiable.** 27B `test_qwen27_paged_engine` 235/235, 35B
   `test_qwen36_paged_engine` 315/315, Qwen3-Coder `test_qwen3coder_paged_engine`
   6/6, Qwen3-dense `test_qwen3_paged_engine` — all UNCHANGED. OPT is new-files;
   any movement means shared behavior was touched.
5. **Build.** Clean CUDA build, `-Werror`, 0 warnings / 0 errors.
6. **memcheck.** 0 errors on the OPT engine path.
7. **Records.** `check-agent-record.py` + `check-doc-checkpoint.py` green.
8. **SPEED — explicitly PENDING** (§7). Not claimed, and the row stays `ACTIVE`
   rather than `DONE` because of it.

### Dependencies

`ENG-RUNNER-MODELSHAPE` (model-shape-agnostic runner — composes with zero
change), `MODEL-FACTORY-registry` (self-registration), the dense BF16 safetensors
loader helpers, the generic paged-attention path. New downward dependency
introduced: `vt::{LayerNorm,Relu,Add}` (this change). No dependency on GDN, MoE,
NVFP4, FP8 or Marlin. Checkpoint dependency: the materialized bf16-safetensors
dir from `scripts/opt-materialize-checkpoint.py` (D2).

### Work breakdown

- **W0** — registry stub + config parse + this spec. `opt_registry.cpp` +
  `opt.h`; registry test 4→5 archs.
- **W1** — reusable-piece extraction: `LoadMergedBf16Vector` in
  `dense_weight_loaders.h`; the three new `vt::` ops with their CPU/CUDA TUs and
  unit test.
- **W2** — `opt_weights.cpp` loader + `test_opt_load.cpp`.
- **W3** — `opt.cpp` forward.
- **W4** — the SACRED gate: oracle capture + determinism measurement, tokenizer
  fixes (L4a/L4b), `test_opt_paged_engine.cpp`, regressions, memcheck, records.
- **W5** — SPEED close vs graphed vLLM. **NOT DONE** (§7).

### Risks/decisions

- **D1 — dtype: run BOTH arms in BF16, not the checkpoint's FP16.**
  facebook/opt-125m ships `torch_dtype: float16`, but `kF16` is unimplemented
  across our CUDA compute path (only `cuda_gdn.cu` handles it), so an fp16-native
  run is impossible today. `--dtype bfloat16` is a first-class vLLM production
  mode, so the oracle was captured under it and our checkpoint materialized bf16
  with the SAME single fp16→bf16 rounding vLLM applies at load. This keeps the
  comparison apples-to-apples and mirrors a real vLLM mode; it does NOT paper
  over anything, but it does leave fp16 coverage genuinely absent (L6).
- **D2 — materialize the checkpoint rather than widen the loader.** The HF
  snapshot is a torch pickle with no `tokenizer.json`. Teaching our loader to
  read pickles is a large, security-sensitive surface with no parity value
  (vLLM itself prefers safetensors), so the conversion is a committed,
  reproducible script (`scripts/opt-materialize-checkpoint.py`) and the tests are
  gated on its output dir. Recorded as gap L5, not as a solved problem.
- **D3 — do NOT force OPT through `dense_attn::AttnBlock`.** Generalizing that
  block with `if (has_rope)` / `if (has_qk_norm)` / `if (has_bias)` flags would
  have made the diff look more additive while making the shared Qwen path more
  fragile and risking the 27B/35B numerics. OPT keeps its own attention block and
  reuses the glue. The duplication is ~40 lines and the gate models are provably
  untouched. Revisit only when a THIRD family wants the same shape.
- **D4 — reject untestable upstream variants loudly.** `word_embed_proj_dim !=
  hidden_size` and non-relu activations are implemented upstream but have no
  checkpoint on the box; `ParseOPTConfig` throws with an explicit message rather
  than shipping an ungated path.
- **R1 — the near-miss that nearly wasn't caught.** The prepended-BOS bug
  produced fluent output and a plausible-looking failure. Committing vLLM's
  `prompt_token_ids` as goldens is what made it a one-line diagnosis. Every
  future cross-family row should capture the oracle's tokenization too.
- **R2 — POST-LN (`do_layer_norm_before=false`, OPT-350m) is implemented but
  UNGATED.** Both branches of `RunLayer` are written from upstream, but only the
  pre-LN branch has a checkpoint. Tracked debt; gate it if OPT-350m is ever
  downloaded.
- **R3 — head_dim 64 is correct but unoptimized.** It falls to the generic paged
  path (S6); the FA2 fast kernels are d128/d256-only. This is the main known
  contributor to the pending speed bar.

---

## 5. Gate result (W4, 2026-07-21)

**vLLM self-determinism (the gate selector):** `scripts/opt-oracle-capture.py
--model ~/models/opt-125m-bf16-st --runs 5` — **all 6 prompts deterministic over
K=5 runs, 0 multi-valued (prompt,pos) cells.** Verdict recorded by the script:
`ALL DETERMINISTIC -> STRICT token-exact gate`.

**Our result: `test_opt_paged_engine` — 6/6 prompts STRICT token-exact,
96/96 tokens, 36/36 assertions.** No near-tie band was used or needed. Per-prompt
tokenization also matches vLLM's `prompt_token_ids` exactly.

This is the "real strict pass on a deterministic model" half of the ratified
ambition ([[near-tie-distributional-gate]]) — achieved on a model from a family
we had never ported.

### 5a. dgx gate series (production flags, one `flock /tmp/gpu`)

Tree transferred by `git archive` (never rsync); goldens md5-verified identical
BEFORE and AFTER the series (`3909e6ac…` / `42b783ac…`, matching the committed
hashes). Configure log confirms CUTLASS sm120a + Marlin NVFP4 MoE + FA2 sm_121a +
vendored Triton AOT. Runner: `scripts/opt-dgx-gate.sh`.

| Gate | Result |
|---|---|
| Clean full CUDA rebuild, `-Werror` | **0 warnings / 0 errors** |
| `test_ops_layernorm` (CUDA leg live) | 5 cases / **29966 assertions** |
| `test_opt_load` | **958 assertions** |
| `test_opt_paged_engine` (SACRED) | **6/6 prompts, 96/96 tokens**, 36 assertions |
| REGRESSION 27B `test_qwen27_paged_engine` | **235/235 UNCHANGED** |
| REGRESSION 35B `test_qwen36_paged_engine` | **315/315 UNCHANGED** |
| REGRESSION `test_qwen3coder_paged_engine` | **6/6 (138 assertions) UNCHANGED** |
| REGRESSION `test_qwen3_paged_engine` | **664 assertions UNCHANGED** |
| REGRESSION `test_model_registry` | 20 cases / 231 assertions |
| REGRESSION `test_pretokenizer` | 18 cases / **97926 assertions** |
| REGRESSION `test_model_loader_gguf` | 3/3 |

**memcheck — stated precisely rather than as a bare "0".**
`compute-sanitizer --tool memcheck --leak-check=full` on the OPT engine path:
**0 invalid reads, 0 invalid writes, 0 invalid `__global__` accesses** (zero
memory errors in the meaningful sense), with the gate still passing 6/6 under the
sanitizer. It does print 66 leak reports (34.4 MB) — these are NOT an OPT leak.
63 of 66 trace to `dense_attn::DBuf`, i.e. the shared `DevicePool`, whose `Put`
returns blocks to a free list and never calls `Backend::Free` when uncapped (cap
0 on GB10, `device_pool.h:65-77`); every pooled block is therefore retained for
the process lifetime BY DESIGN and is necessarily reported at exit. Measured
against the pre-existing baseline on the SAME binary:

| Binary under memcheck | invalid accesses | leak reports | from `DBuf` | bytes |
|---|---|---|---|---|
| `test_opt_paged_engine` (NEW) | **0** | 66 | 63 | 34.4 MB |
| `test_qwen3_paged_engine -tc=qwen3-0.6B*` (pre-existing) | **0** | 125 | 98 | 35.9 MB |
| `test_qwen3_paged_engine` full (pre-existing) | **0** | 184 | all | 39.7 MB |

OPT retains FEWER pool blocks than the dense model that preceded it — same class
as the retained allocations already documented for the Qwen3-Coder gate.

**CPU suite:** `-Werror` 0-warn; full CTest 136/138, the two misses being
`test_openai_api_server` + `test_openai_conformance`, which PASS re-run isolated
(2/2) — the documented `-j` port-contention flake for that suite, not a
regression from this change.

---

## 6. Diagnosis log (what broke, and how it was found)

Recorded because the bring-up's value is partly the failure modes it exposed:

| Symptom | Root cause | Fix |
|---|---|---|
| `ByteLevel pre-tokenizer with use_regex=true unsupported` at engine load | no GPT-2 byte-level split (L4a) | `SplitPattern::kGpt2` + `PretokenizeGpt2` |
| `vt: cast_bf16: in must be f32` in the attention block | used `CastBf16` to compact strided qkv slices | use `vt::QkvSplit` (dense per-shard outputs) |
| **0/6 prompts, fluent but wrong text**; `prompt_token_ids` `{1121,…}` vs vLLM `{2,1121,…}` | post-processor BOS parsed but never applied (L4b) | `EncodeWithSpecialTokens` + 1 line in `input_processor.cpp` → **6/6** |
| loader test `rotary_dim == 0` failed with 64 | `HfConfig` synthesizes RoPE fields unconditionally (L3) | assertion corrected to pin and document the leak |

---

## 7. SPEED — explicitly PENDING

No throughput measurement has been taken and none is claimed. The row is
`ACTIVE`, not `DONE`. Known structural gaps before a speed run is worth doing:
head_dim 64 falls to the generic paged path rather than FA2 (R3/S6); there is no
decode CUDA graph for OPT (the Qwen3-Coder W7 pattern is the template); and the
new LayerNorm/Add/Relu kernels are plain correctness-grade grid-stride launches
with no fusion into the surrounding GEMMs. Per
[[benchmark-gate-statistics]] and the acceptance rule, a speed claim requires the
full every-axis grid vs a graphed vLLM denominator on an idle box; that is a
separate increment.
