# Spec: MTP speculative decoding (task #51, from B5 scoping)

Implementation-ready spec for Multi-Token-Prediction speculative decoding on
both gate checkpoints. Derived from
[.agents/specs/spec-decode-scoping-2026-07-10.md](spec-decode-scoping-2026-07-10.md);
every claim re-grounded against the pin (`/home/mudler/_git/vllm` @ `e24d1b24`)
and the actual checkpoints on `dgx.casa`. Companion spec:
[dflash-spec-decode.md](dflash-spec-decode.md) (depends on this one).

## Protocol compliance map

| Required field | Grounded content |
|---|---|
| Row IDs | `SPEC-MTP`, `SPEC-REJECTION`, `SPEC-GDN-SEGMENTS`, `MODEL-SPEC-qwen3-5-mtp-qwen3-5-mtp`, `MODEL-SPEC-qwen3-5-mtp-qwen3-5-moe-mtp` |
| Scope | k=1 Qwen3.5/3.6 MTP first, both gate models and GDN rollback; §0-§3. Grammar bitmask under spec decode (multi-row bitmask over draft tokens): OUT of scope — deferred, see porting-inventory §6 |
| Upstream chain | config, model, speculator, rejection, scheduler and KV/state paths at pin; §2-§3 |
| Our baseline | exact local GDN, scheduler, runner, model-loader and state touch points; §3.1 and §4 |
| Port map | upstream component to the local file set; §3.1, §4 and the milestone file lists in §5 |
| Tests to port | exact upstream modules/cases and local tiers; §6 |
| Gates | phase correctness, greedy acceptance, memory and same-workload throughput gates; §5 |
| Dependencies | scheduler/KV/model loader, sampler/rejection, GDN rollback, both gate-model/quant rows; §3-§5 |
| Work breakdown | non-overlapping `M-mtp-0` through `M-mtp-4` claims; §5 and §7 |
| Risks/decisions | checkpoint depth resolution, GDN slot rollback, graph/state sizing and acceptance semantics; §0, §3 and §8 |

The three engine rows may be implemented by separate claims after their shared
metadata ABI is frozen. The two model rows own only their target-specific MTP
weight/config mapping; they do not implicitly own the rejection or GDN rows.

## 0. CORRECTION to B5 (changes the route's shape, not its order)

B5 §2/§4 claimed the 27B is "pure attention" with "no GDN problem → port there
first". **That is factually wrong.** Both gate checkpoints are GDN hybrids:

- 27B config (`unsloth--Qwen3.6-27B-NVFP4` snapshot `890bdef`):
  `text_config.layer_types` = 64 entries, 3-of-4 `linear_attention`,
  `full_attention_interval: 4` → **48 GDN layers + 16 full-attn layers**.
- 35B config (`nvidia--Qwen3.6-35B-A3B-NVFP4` snapshot `491c2f1`): 40 layers,
  30 GDN + 10 full-attn (same interval).
- Upstream `vllm/model_executor/models/qwen3_5.py:129-136` instantiates
  `QwenGatedDeltaNetAttention` for `linear_attention` layers in the 27B model
  class; our own `include/vllm/model_executor/models/qwen3_5_dense.h:58-62`
  (`is_linear_attention`, `GdnLayerWeights`) mirrors that.

**Consequence:** the GDN spec path (§4) is on the critical path of milestone 1
— it cannot be deferred to the 35B. The order (27B first) still stands: the
27B avoids only the MoE-MTP-layer complexity, not the GDN one. The good news
(§4.5): our CUDA GDN decode kernel already has the right shape for the
upstream mechanism, so this is an incremental kernel extension, not a rewrite.

## 1. What ships in the checkpoints (verified on disk, dgx.casa)

Both NVFP4 gate checkpoints ship a 1-layer MTP head, **all BF16** (never
quantized). `text_config.mtp_num_hidden_layers: 1`,
`mtp_use_dedicated_embeddings: false` (both configs) → the head shares the
target's `embed_tokens` AND `lm_head` (`tie_word_embeddings: false`, vocab
248320).

### 27B — 15 `mtp.*` tensors (single `model.safetensors`, snapshot `890bdef7`)

| tensor | dtype | shape |
|---|---|---|
| `mtp.fc.weight` | BF16 | [5120, 10240] |
| `mtp.pre_fc_norm_embedding.weight` | BF16 | [5120] |
| `mtp.pre_fc_norm_hidden.weight` | BF16 | [5120] |
| `mtp.layers.0.input_layernorm.weight` | BF16 | [5120] |
| `mtp.layers.0.self_attn.q_proj.weight` | BF16 | [12288, 5120] |
| `mtp.layers.0.self_attn.k_proj.weight` | BF16 | [1024, 5120] |
| `mtp.layers.0.self_attn.v_proj.weight` | BF16 | [1024, 5120] |
| `mtp.layers.0.self_attn.o_proj.weight` | BF16 | [5120, 6144] |
| `mtp.layers.0.self_attn.q_norm.weight` | BF16 | [256] |
| `mtp.layers.0.self_attn.k_norm.weight` | BF16 | [256] |
| `mtp.layers.0.post_attention_layernorm.weight` | BF16 | [5120] |
| `mtp.layers.0.mlp.gate_proj.weight` | BF16 | [17408, 5120] |
| `mtp.layers.0.mlp.up_proj.weight` | BF16 | [17408, 5120] |
| `mtp.layers.0.mlp.down_proj.weight` | BF16 | [5120, 17408] |
| `mtp.norm.weight` | BF16 | [5120] |

Shape decode: head_dim 256, 24 q-heads, 4 kv-heads; `q_proj` out 12288 =
2×24×256 because Qwen3.5 full attention is **output-gated**
(`attn_output_gate` defaults True, `qwen3_next.py:258-263` — q_proj packs
q(6144)+gate(6144)); `o_proj` in 6144 = 24×256. Partial rotary 0.25,
rope_theta 1e7 — same as the target's full-attn layers.

### 35B — 19 `mtp.*` tensors (all in `model-00003-of-00003.safetensors`, snapshot `491c2f1e`)

| tensor | dtype | shape |
|---|---|---|
| `mtp.fc.weight` | BF16 | [2048, 4096] |
| `mtp.pre_fc_norm_embedding.weight` / `_hidden.weight` | BF16 | [2048] |
| `mtp.layers.0.input_layernorm.weight` | BF16 | [2048] |
| `mtp.layers.0.self_attn.{q,k,v,o}_proj.weight` | BF16 | q [8192,2048], k/v [512,2048], o [2048,4096] |
| `mtp.layers.0.self_attn.{q,k}_norm.weight` | BF16 | [256] |
| `mtp.layers.0.post_attention_layernorm.weight` | BF16 | [2048] |
| `mtp.layers.0.mlp.gate.weight` | BF16 | [256, 2048] |
| `mtp.layers.0.mlp.experts.gate_up_proj` | BF16 | [256, 1024, 2048] |
| `mtp.layers.0.mlp.experts.down_proj` | BF16 | [256, 2048, 512] |
| `mtp.layers.0.mlp.shared_expert.{gate,up}_proj.weight` | BF16 | [512, 2048] |
| `mtp.layers.0.mlp.shared_expert.down_proj.weight` | BF16 | [2048, 512] |
| `mtp.layers.0.mlp.shared_expert_gate.weight` | BF16 | [1, 2048] |
| `mtp.norm.weight` | BF16 | [2048] |

Same gated attention (16 q-heads ×256 ×2, 2 kv-heads). The MTP MLP is the
**full MoE**: 256 routed experts (top-8) + shared expert — i.e. the 35B MTP
layer = one complete target-style full-attention MoE decoder layer, in BF16.

**Key fact for both:** the MTP layer is `layer_type="full_attention"` — the
DRAFT never runs GDN (`qwen3_5_mtp.py:105-112` hardcodes
`Qwen3_5DecoderLayer(layer_type="full_attention")`). The GDN problem lives
only in the TARGET's verify step (§4).

## 2. vLLM V1 anatomy at pin (the 1:1 mirror target)

We mirror the MRV2 stack **`vllm/v1/worker/gpu/spec_decode/`** (NOT legacy
`vllm/v1/spec_decode/`). Caveat recorded: at this pin, hybrid (GDN) models are
not default-routed to MRV2 (`vllm/config/vllm.py:565-566` `is_hybrid → False`)
and the e2e test skips Qwen3.5+MRV2
(`tests/v1/e2e/spec_decode/test_spec_decode.py:816-820` "Model Runner V2 does
not yet support hybrid models") — but the MRV2 tree contains the complete
hybrid+spec plumbing (`model_states/mamba_hybrid.py`) and the GDN
backend/kernels are runner-agnostic (`vllm/v1/attention/backends/gdn_attn.py`,
`fla/ops/*` — shared with the V1 runner that upstream actually validates on).
We mirror MRV2 structure and validate token-exactness against the vLLM oracle
in its default (V1-runner) config, same as our existing gates.

### 2.1 Config resolution
- `vllm/config/speculative.py:480-489`: `model_type qwen3_5|qwen3_5_moe` →
  `qwen3_5_mtp`, arch `Qwen3_5MTP` / `Qwen3_5MoeMTP`, `n_predict =
  mtp_num_hidden_layers` (=1). With `--speculative-config
  '{"method":"mtp"}'` and no explicit k, `num_speculative_tokens` defaults to
  `n_predict` = **1** (`speculative.py:865-875`) — so k=1 IS vLLM's default
  for these checkpoints.
- Method dispatch: `spec_decode/__init__.py:29-31` → `MTPSpeculator`.

### 2.2 Speculator classes
- `speculator.py:31` `BaseSpeculator` (capture/propose interface),
  `:74` `DraftModelSpeculator` (buffers: `draft_tokens [max_reqs, k]`,
  per-req temperature/seeds/idx_mapping; `_greedy_sample_draft` = argmax
  `:255-259`; draft sampling ignores top-k/top-p by design, temperature only —
  `:298-303` — does NOT change the output distribution after rejection).
- `autoregressive/speculator.py:30` `AutoRegressiveSpeculator`:
  - `propose()` `:127-271`. Draft **prefill reuses the target's attention
    metadata + slot mappings** unchanged (`:222-234`) because batch shape and
    KV layout are identical (rejected positions are padded, not compacted —
    NOTE `:162-167`).
  - `prepare_prefill_inputs` Triton kernel `:469-588`: shift target input_ids
    left by 1 within each request's query span; splice the next token at the
    last position (`last_sampled` if the request sampled, else
    `next_prefill_tokens` for chunked prefill `:502-508`); `query_len -=
    num_rejected` (`:499-500`).
  - k=1 **early-exits after one forward** (`:236-238`) — no multi-step decode,
    no draft-side decode attn metadata build. This is why k=1 is the right
    first milestone: it exercises all the shared plumbing with the minimum
    kernel surface.
  - k>1: `prepare_decode_inputs` `:591-665` + `_multi_step_decode` `:371-418`
    (k−1 single-token decode steps, `update_draft_inputs` `:668-765`).
- `mtp/speculator.py:12-18`: `MTPSpeculator` is an **18-line subclass** — only
  `load_draft_model` = `load_eagle_model`.
- `eagle/utils.py:29-80` `load_eagle_model`: loads the draft model
  (`Qwen3_5MTP`) then **shares the target's `embed_tokens` and `lm_head`**
  when the draft has no own copy (our checkpoints: it never does).

### 2.3 The MTP model (`vllm/model_executor/models/qwen3_5_mtp.py`)
Forward (`:129-165`):
```
h = fc( cat[ pre_fc_norm_embedding(embed(input_ids)),
             pre_fc_norm_hidden(target_hidden) ] )   # [T,2H] -> [T,H]
h = decoder_layer(positions, h)                      # full-attn (+MoE on 35B)
h = norm(h)                                          # -> lm_head (shared)
```
- `target_hidden` = the target model forward's OUTPUT hidden states, i.e.
  post-final-norm (`qwen3_5.py:339-351` returns `self.model(...)` output).
- **NVFP4 gotcha** (`:86-103`): `mtp.fc` is stored BF16 but missing from the
  quant-config exclude list → vLLM forces `fc` unquantized when quant is
  `modelopt_fp4`. Our loader must treat ALL `mtp.*` as bf16-unquantized (our
  27B loader already declares this: `qwen3_5_dense_weights.cpp:210`).
- Weight names remap `mtp.` → `model.` (`:282-295`); the draft's KV layer is
  registered as a NEW attention layer (index `num_hidden_layers`, i.e. layer
  64/40) → it gets its **own KV-cache layer** (draft layer names = all minus
  target's, `speculator.py:163-169`).

### 2.4 Rejection sampling (`spec_decode/rejection_sampler.py`, `rejection_sampler_utils.py`)
- Verify-step logits: the target runs the k+1-token query per request;
  `input_batch.logits_indices` covers ALL spec positions (not just last):
  `input_batch.py:91-93,146-148` (`cu_num_logits` per request = 1+k).
- `RejectionSampler.__call__` `:101-160`: apply sampling params on the
  expanded batch, then `rejection_sample` (`rejection_sampler_utils.py:864`).
- **Greedy path** (all we need for milestone 1): `_rejection_kernel`
  `rejection_sampler_utils.py:524,564-585` — `is_greedy = temp == 0`; accept
  iff `draft_sampled == target_argmax`; on first mismatch emit the target
  argmax as the bonus/replacement token. k=1 greedy verify = argmax over
  `[num_reqs, 2, vocab]` + equality.
- Stochastic path (milestone 4): Gumbel-based, seeds keyed by position
  (`speculator.py:273-285` draft uses `positions + 1` to match verify-side
  noise); block verification variant `:535+`.
- Output: `num_sampled`/`num_rejected` per request
  (`input_batch.py get_num_sampled_and_rejected`, used at
  `rejection_sampler.py:146-152`).

### 2.5 Scheduler + engine-core integration
- `num_lookahead_tokens`: `sched/scheduler.py:245-252` — MTP
  (`uses_draft_model()` incl. "mtp" `speculative.py:1185`) sets it to k.
  Flows into `allocate_slots` so verify slots exist ahead of time.
- Draft tokens flow: runner returns them OUT-OF-BAND —
  `model_runner.py:1483-1489` `draft_tokens_handler.set_draft_tokens` →
  `take_draft_token_ids` → engine core `v1/engine/core.py:515-517` →
  `scheduler.update_draft_token_ids` (`scheduler.py:1937`) sets
  `request.spec_token_ids` (async path: `:1959`).
- Scheduling a spec request: `scheduler.py:593-609` — schedules
  `1 + len(spec_token_ids)` tokens (`num_tokens_with_spec`), records
  `scheduled_spec_decode_tokens[req_id]`, clears `request.spec_token_ids`.
- Rejection rollback: `scheduler.py:1580-1612` — `num_accepted =
  len(generated) − 1`, `num_rejected = num_draft − num_accepted`,
  `request.num_computed_tokens -= num_rejected` (+ same for
  `num_output_placeholders` under async). **Paged KV needs nothing else**
  (the slots simply get overwritten); GDN state is §4.
- Runner: `model_runner.py:1065-1079` routes sampling through
  `RejectionSampler` when `input_batch.num_draft_tokens > 0`; `:1455-1481`
  calls `speculator.propose(...)` after sampling, feeding `num_sampled`,
  `num_rejected`, `last_sampled`, `next_prefill_tokens`.

### 2.6 KV-cache sizing for the draft layer + GDN spec slots
- Draft full-attn KV layer: one more `FullAttentionSpec` layer — normal paged
  allocation, nothing special.
- **GDN/Mamba layers get k extra state slots per request**:
  `mamba/abstract.py:55-59` sets `MambaSpec.num_speculative_blocks =
  num_speculative_tokens`; `kv_cache_interface.py:675,692-697` sizes pages;
  `single_type_kv_cache_manager.py:1206-1215` allocates the extra blocks
  (non-"align" mamba cache mode; **"all" prefix-caching mode is rejected** for
  Qwen3_5MTP, `qwen3_5_mtp.py:206-210` — mirror that check).

### 2.7 FROZEN spec-metadata ABI (SPEC-MTP **I2 scheduler-half, LANDED**)

I2 landed the host-side scheduler/engine plumbing and **FROZE** the metadata ABI
the remaining increments build against. I3 (rejection sampler) and I5 (verify /
propose runner) code against exactly these declarations — they are stable, and
every one of them is **inert on the production default path** (no
`SpeculativeConfig` ⇒ `num_lookahead_tokens == 0` ⇒ nothing below is ever
populated, so the engine is byte-identical to the pre-spec engine).

| # | Declaration | Where | Mirrors @ `e24d1b24` |
|---|---|---|---|
| 1 | `struct SpeculativeConfig { std::string method; std::optional<int> num_speculative_tokens; int n_predict; }` + `ResolveMtp(mtp_num_hidden_layers, user_k=nullopt)`, `ResolvedNumSpeculativeTokens()`, `use_eagle()`, `uses_draft_model()`, `use_dflash()`, `NumLookaheadTokens()` | `include/vllm/config/speculative.h` (NEW) | `vllm/config/speculative.py:480-489,865-875,1163-1195`; lookahead rule `sched/scheduler.py:275-292` |
| 2 | `struct DraftTokenIds { std::vector<std::string> req_ids; std::vector<std::vector<int32_t>> draft_token_ids; }` | `include/vllm/v1/engine/types.h` | `vllm/v1/outputs.py:310-315` |
| 3 | `std::vector<int32_t> Request::spec_token_ids` + `int Request::NumTokensWithSpec()` | `include/vllm/v1/request.h` | `vllm/v1/request.py:152,251-252` |
| 4 | `SchedulerOutput::scheduled_spec_decode_tokens` (`map<string, vector<int32_t>>`) — field ALREADY existed; I2 is the first code to POPULATE it | `include/vllm/v1/core/sched/output.h:147` (unchanged) | `sched/scheduler.py:593-609` |
| 5 | `Scheduler(..., std::optional<SpeculativeConfig> speculative_config = std::nullopt)` + `void Scheduler::update_draft_token_ids(const DraftTokenIds&)` | `include/vllm/v1/core/sched/scheduler.h` | `sched/scheduler.py:275-292,1937-1957` |
| 6 | `virtual std::optional<DraftTokenIds> ModelRunnerBase::take_draft_token_ids()` (default `nullopt`) and the `Executor::take_draft_token_ids()` pass-through | `include/vllm/v1/worker/gpu/model_runner_base.h`, `include/vllm/v1/executor/executor.h` | `model_runner.py:1483-1489`; `executor/abstract.py` |
| 7 | `EngineCore(..., bool check_for_draft_tokens = false)` + `void EngineCore::post_step(bool model_executed)` | `include/vllm/v1/engine/core.h` | `v1/engine/core.py:186-190,509-517` |
| 8 | `InputBatch::num_accepted_tokens` (per-slot, seeded to 1) + `InputBatch::update_req_spec_token_ids(req_index, req_id, scheduled_spec_tokens)`; `spec_token_ids` per-slot list now populated | `include/vllm/v1/worker/gpu/input_batch.h` | `gpu_input_batch.py:240-243,286,484-509` |

**Behavior landed** (all gated behind an actually-configured speculator):
`num_tokens_with_spec` in the running-loop budget; a running request carrying
drafts is scheduled for `1 + len(spec_token_ids)` tokens and recorded in
`scheduled_spec_decode_tokens` (prefix-clamped when the budget fits only part
of the drafts), then `spec_token_ids` is cleared; the `num_computed_tokens`
rollback by `num_rejected` on `update_from_output`; and
`update_draft_token_ids` installing fresh drafts (skipping unknown/finished
requests, and clearing rather than installing for a request still in a prefill
chunk).

### 2.8 The GREEDY ACCEPT RULE, as implemented (SPEC-REJECTION **I3, LANDED**)

I3 landed the VERIFY half against the §2.7 ABI: the input-batch logits expansion
plus the greedy rejection sampler. It is likewise INERT on the production
default (no `SpeculativeConfig` ⇒ the scheduler never populates
`scheduled_spec_decode_tokens` ⇒ `num_draft_tokens == 0` on every step).

**Logits expansion.** `prepare_inputs` (`src/vllm/v1/worker/gpu/prepare_inputs.cpp`)
now emits `StepInputs::cu_num_logits` `[num_reqs+1]`,
`num_draft_tokens_per_req` `[num_reqs]` and `num_draft_tokens`, and widens
`logits_indices` from `[num_reqs]` to `[Σ(1+k_i)]`. Mirrors
`gpu/model_runner.py:866-898` (the `if not draft_tokens:` / else split) and the
index formula of `_combine_sampled_and_draft_tokens_kernel`
(`gpu/input_batch.py:317-327`): per request `logits_start = query_end -
(1 + k_i)`, then `1 + k_i` consecutive rows — the `k_i` draft positions followed
by the bonus position. **With no drafts the else-branch is not taken:**
`cu_num_logits = arange(num_reqs+1)` and `logits_indices =
query_start_loc[1:] - 1`, the pre-I3 array, byte for byte.

**The accept rule (the whole correctness story for the M-mtp-1 e2e gate).**
Mirrors the `is_greedy` branch of `_rejection_kernel`
(`rejection_sampler_utils.py:564-585`, accepted-length store at `:628`), the
greedy short-circuit of `_resample_kernel` (`:846-849`) and
`_insert_resampled_kernel` (`:828-841`). For request `r` over rows
`[cu[r], cu[r+1])` with `k_r = cu[r+1] - cu[r] - 1`:

```
accepted = true; len = 0
for i in [0, k_r):
  if !accepted: break                              # upstream `elif accepted:`
  target_argmax = argmax(logits[cu[r] + i])        # lowest-index tie-break
  draft         = draft_sampled[cu[r] + i + 1]     # NOTE the +1, :534
  accepted     &= (draft == target_argmax)
  sampled[r][i] = accepted ? draft : target_argmax # replacement on mismatch
  len          += accepted
if len == k_r: sampled[r][k_r] = argmax(logits[cu[r] + k_r])   # BONUS
num_sampled[r]  = len + 1
num_rejected[r] = (1 + k_r) - num_sampled[r] = k_r - len
```

`draft_sampled` is `input_ids[logits_indices]` (`rejection_sampler.py:111`).
`num_sampled` feeds `InputBatch::num_accepted_tokens` (as `max(num_sampled, 1)`,
`model_states/mamba_hybrid.py:290-310`) and `num_rejected` feeds I2's
`num_computed_tokens -= num_rejected` rollback. A `-1` placeholder draft is
rejected by construction (an argmax is `>= 0`), with no out-of-bounds read.
**`k_r == 0` therefore emits exactly one token — the plain greedy argmax — which
is the byte-identity anchor for the default path.**

**Code.** `include/vllm/v1/spec_decode/rejection_sampler.h` +
`src/vllm/v1/spec_decode/rejection_sampler.cpp` (the host `RejectionSampler`,
`RejectionSamplerOutput{sampled_token_ids, num_sampled, num_rejected}`, and the
`get_num_sampled_and_rejected` chunked-prefill zeroing,
`gpu/input_batch.py:408-453`); one additive vt op `kGreedyRejectionSample` /
`vt::GreedyRejectionSample` with a CPU reference (`src/vt/cpu/cpu_sample.cpp`)
and a CUDA mirror of upstream's two-phase decomposition
(`src/vt/cuda/cuda_sample.cu`: `RejectionRowArgmaxKernel` per expanded row, then
a one-thread-per-request `GreedyRejectAcceptKernel`); routing at
`GPUModelRunner::sample_tokens` gated on `num_draft_tokens > 0`
(`gpu/model_runner.py:1065-1077`).

**Tests ported** (`tests/vllm/v1/spec_decode/test_rejection_sampler.cpp`):
`test_perfect_match`, `test_early_mismatch`, `test_multiple_sequences`,
`test_single_token_sequence`, `test_empty_sequence`, `test_multiple_mismatches`,
`test_parametrized_cases` (`tests/v1/sample/test_rejection_sampler.py`),
`test_greedy_rejection_sample` k∈{1,3} and
`test_placeholder_draft_token_rejected`
(`tests/v1/spec_decode/test_rejection_sampler_utils.py`), plus the expansion
cases in `tests/vllm/v1/worker/test_prepare_inputs.cpp` and CUDA==CPU
bit-exactness at the gate vocab 248320 in `tests/vt/test_cuda_ops.cpp`.

**STILL DEFERRED after I3** (the obvious seams): the stochastic/Gumbel accept
(`:589-627`) and residual resample (`:775-799`); block verification (`:535+`);
synthetic acceptance rates; `apply_sampling_params` over the EXPANDED batch
(`rejection_sampler.py:113-120`, needs `expanded_idx_mapping` /
`expanded_local_pos` — greedy argmax is invariant under temperature/top-k/top-p,
so this is a no-op for the greedy gate but REQUIRED before penalties /
logit-bias / bad-words work under spec decode); expanded-batch logprobs
(`rejection_sampler.py:67-99`); and the multi-row grammar bitmask.

**STILL DEFERRED** (unchanged by I2, called out so I3/I5 do not assume them):
the first-decode-step spec **padding** (placeholder `-1` drafts /
`num_spec_tokens_to_schedule` / `num_invalid_spec_tokens`), the dynamic-SD
lookup, the async draft-in-output path
(`update_draft_token_ids_in_output`, `scheduler.py:1959`), and the
structured-output grammar validation of proposed drafts.

## 3. GDN linear-state rollback — the mechanism (B5's hard problem, answered)

Linear-attention state is not a paged KV you can truncate. Upstream solves
rollback by **never rolling back: it writes a state snapshot per draft
position into per-request slot arrays, and the NEXT step selects its initial
state by acceptance count.** No copies, no undo.

1. **Slots**: each spec-decoding request has `num_spec+1` mamba state slots —
   `spec_state_indices_tensor[i, :] = block_table_row[: num_spec+1]`
   (`gdn_attn.py:127-131` shape, `:267-269`/`:288-290` fill; backed by the §2.6
   extra blocks).
2. **Write**: during the verify forward, the GDN recurrent kernel processes
   the k+1 query tokens sequentially and stores the post-token state of
   timestep `t` into slot `spec_state_indices[i, t]`
   (`fla/ops/fused_sigmoid_gating.py:156-166`, `INPLACE_FINAL_STATE` branch).
3. **Read**: the initial state for the step is loaded from slot
   `num_accepted_tokens[i] − 1` — the snapshot after the last ACCEPTED token
   of the previous step (`fused_sigmoid_gating.py:103-116`,
   `IS_SPEC_DECODING`). A rejection therefore just selects an earlier slot.
4. **Conv state** (the 4-tap causal conv preceding the GDN): same trick via
   `causal_conv1d_update(..., num_accepted_tokens=..., max_query_len =
   spec_state_indices.size(-1))` (`qwen_gdn_linear_attn.py:1344-1356`).
5. **Acceptance plumbing** (runner side): `model_states/mamba_hybrid.py:79-81`
   keeps `num_accepted_tokens_gpu [max_reqs]`; `postprocess_state:290-310`
   scatters `max(num_sampled,1)` into it after every step (no host sync);
   `prepare_attn:247-264` gathers it per-batch + builds
   `num_decode_draft_tokens_cpu` (−1 sentinel = non-spec row) for the builder.
6. **Metadata split** (`gdn_attn.py build():189-326`): rows partition into
   spec-decode vs non-spec; **non-spec decodes are reclassified as prefills
   when spec rows exist** (`:243-251` — the chunked prefill kernel handles
   len-1 sequences with initial state exactly, so the decode kernel only ever
   sees uniform spec batches); spec rows run the recurrent kernel with
   `spec_query_start_loc`/`spec_token_indx` gather-order
   (`qwen_gdn_linear_attn.py:1455-1476`).
7. Full-attn layers of the target need NOTHING new: the verify query is just a
   `num_tokens > num_reqs` varlen batch, which our paged-attn prefill kernels
   already route (`cuda_paged_attn.cu:2317`).

### 3.1 The C++ mirror (why this is incremental for us)
- Our GDN state cache is already **slot-indexed**: `GdnStateCache { ssm_state
  [num_state_blocks, Hv, Dv, Dk] f32; conv_state [num_state_blocks, conv_dim,
  K-1] f32 }` (`include/vllm/model_executor/models/qwen3_5.h:67-70`), and our
  fused decode kernel already ports `fused_sigmoid_gating.py` and takes a
  per-request `state_idx` (`src/vt/cuda/cuda_gdn.cu:721` header cite, `:813`
  `GdnDecodeFusedKernel`, `:901-956` launchers).
- Extension: add a `T>1` inner loop + `IS_SPEC` mode to the kernel — initial
  slot = `state_idx[i, num_accepted[i]−1]`, per-timestep store to
  `state_idx[i, t]` — mirroring `fused_sigmoid_gating.py:103-170` 1:1. Same
  for our conv-update kernel (mirror `causal_conv1d.py` `num_accepted_tokens`
  handling).
- Our metadata builder mirrors `build()` minus spec today
  (`src/vllm/v1/attention/backends/gdn_attn.cpp:89` hardwires
  `num_spec_decodes = 0`; fields exist as DEFERRED in
  `include/vllm/v1/attention/backends/gdn_attn.h:74-79`): port the spec
  branches of `gdn_attn.py:189-326` including the decode→prefill
  reclassification and the CPU-side split (no GPU sync).
- Allocation: mirror `num_speculative_blocks` in our mamba single-type manager
  (our `single_type_kv_cache_manager.h` counterpart) so each request's GDN
  block-table row has k+1 slots.
- **Memory cost** (flag, measure at impl): one GDN state slot =
  Hv×Dv×Dk×4B, both models Dk=Dv=128 (`linear_{key,value}_head_dim: 128`).
  27B: 48 heads → 3.0 MiB/slot/layer × 48 GDN layers ≈ **144 MiB per extra
  slot per request**; 35B: 32 heads → 2.0 MiB × 30 layers ≈ **60 MiB**. k=1
  doubles GDN state memory vs today. (This is inherent to vLLM's scheme too —
  we mirror it; at DFlash's k=8/16 it becomes a first-order sizing input, see
  the DFlash spec §5.)

### 3.2 SPEC-GDN-SEGMENTS I4 — LANDED (2026-07-24)

I4 lands the whole §3 mechanism against the §2.7 frozen ABI (consumes
`InputBatch::num_accepted_tokens` and `SchedulerOutput::scheduled_spec_decode_tokens`,
defines neither). It is DEFAULT-OFF and INERT: the builder's `num_spec` is 0
without a `SpeculativeConfig`, so `spec_sequence_masks` is never populated and
`num_spec_decodes` stays 0. **No shipped kernel gained a branch** — both spec
kernels are NEW op ids (`kGdnSpecDecode`, `kCausalConv1dSpecUpdate`) with their
own entry points, so `GdnDecode`/`GdnPackedDecode`/`CausalConv1dUpdate`/`GdnPrefill`
are untouched object code.

**Metadata split + reclassification** — `GDNAttentionMetadataBuilder`
(`src/vllm/v1/attention/backends/gdn_attn.cpp`) gains `num_spec`/`use_full_cuda_graph`
and a `build()` overload taking `num_accepted_tokens` + `num_decode_draft_tokens`
(−1 sentinel = non-spec row). The 3-argument override now DELEGATES to it with
null spec arguments, so there is exactly ONE implementation. Mirrors
`gdn_attn.py:189-326`: spec/non-spec partition, the decode→prefill
reclassification (`:243-251`, the #34845 crash), the k+1-column
`spec_state_indices_tensor`, `spec_query_start_loc`, the stable-sorted
`spec_token_indx`/`non_spec_token_indx`, `spec_sequence_masks`, and the
mask-filtered `num_accepted_tokens`; the FULL-cudagraph padding (`:413-462`) pads
the REQUEST-indexed metadata to `num_reqs` (never the token count) with NULL
slots and accepted=1. **RECORDED DEVIATION:** upstream's NULL sentinel is block 0
(`state_idx <= 0` skips); our GDN cache has no reserved row (slot 0 is live), so
the local ABI uses a NEGATIVE sentinel (`kNullStateSlot = -1`), matching the
existing `vt::GdnDecode`/`GdnPackedDecode` cache ABI.

**GDN `T>1`/`IS_SPEC` recurrence** — new `vt::GdnSpecDecode` (CPU reference +
CUDA), ports `fla/ops/fused_sigmoid_gating.py:66-72,103-116,156-166`. The CUDA
kernel reuses `GdnDecodeFusedKernel`'s geometry and per-token arithmetic verbatim
and adds only the initial-slot select (column `num_accepted[i]-1`) and the
per-timestep snapshot write-back (to column `t`). **BIT-EXACT reference:** a T==1
/ accepted==1 spec call reproduces `vt::GdnDecode` bit-for-bit (memcmp), and every
per-timestep snapshot slot j is memcmp-identical to a chain of shipped
`vt::GdnDecode` calls over only tokens 0..j — asserted at the real 27B (Hv=48) and
35B (Hv=32) GDN dims, Dk=Dv=128, on CPU and CUDA.

**Conv state honouring `num_accepted_tokens`** — new `vt::CausalConv1dSpecUpdate`,
ports `causal_conv1d.py:818-1067` + the wrapper's `state_len = width-1 + (seqlen-1)`
(`:1181-1184`). The conv row widens to `(K-1)+k` taps; it is read at offset
`num_accepted-1` and shifted left by exactly ONE tap per step, so the window
advances by the ACCEPTED count, not by `1+k`. The conv needs no extra SLOTS, only
column 0 of the spec index row (`qwen_gdn_linear_attn.py:1350`).

**Allocation** — `MakeQwen3_5KVCacheSpec(config, block_size, num_blocks, num_spec)`
sets the conv row to `conv_kernel-1+num_spec` (`mamba_utils.py:226`) and
`MambaSpec::num_speculative_blocks = num_spec` (`mamba/abstract.py:55-59`),
driving the already-present `MambaManager` k+1 block allocation. The registered
`KVCacheSpecBuilder` delegates with `num_spec=0`, so the production spec is
byte-unchanged.

**MEASURED state-memory cost** (from the real dims; pinned by
`test_model_registry`): one f32 SSM slot is `Hv·Dv·Dk·4B`. **35B** (Hv=32):
32·128·128·4 = **2.0 MiB/slot/layer**, ×30 GDN layers = **60 MiB per extra slot
per request**; **27B** (Hv=48): **3.0 MiB/slot/layer**, ×48 GDN layers = **144
MiB per extra slot per request**. `k=1` allocates one extra slot, so it DOUBLES
the GDN SSM state per in-flight request; the conv row grows by only
`conv_dim·sizeof(bf16)` per block. On GB10's 119 GiB unified pool this is a
per-concurrency multiplier on the GDN state, not on the weights — at decode
concurrency 16 the 27B extra SSM state is ~16·144 MiB ≈ 2.25 GiB, within
headroom; it becomes a first-order sizing input only at DFlash k=8/16 (DFlash
spec §5).

**Tests.** `test_gdn_metadata_builder` (20 cases / 483 assertions) grows the whole
upstream `GDN_BUILD_TEST_CASES` table + reclassification + CG-padding cases +
the default-off byte-identity proof; `test_ops_gdn` grows the reject-at-every-j
rollback proof (CPU + CUDA, real dims) for both the SSM recurrence and the conv
window; `test_model_registry` pins the k+1 slot / widened-conv sizing and the
`num_spec==0` byte-identity. RED-first proven by four reverted experiments (the
hardwired-0 metadata, the wrong-snapshot-slot, the ignore-accepted read, and the
1+k conv advance all fail).

## 4. Our-code touch points (survey, file:line)

| area | where | change |
|---|---|---|
| loader 27B | `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp` (`:209` already exempts `mtp.*` from quant) + `qwen3_5_dense.h:101` (DEFERRED note) | add `MtpWeights` struct + load the 15 tensors (bf16); share embed/lm_head refs |
| loader 35B | `src/vllm/model_executor/models/qwen3_5_weights.cpp:328-334` ("mtp.* … not requested") | request + load the 19 tensors incl. MoE expert stacks (bf16) |
| GGUF | `qwen3_5_gguf_weights.cpp` | mtp.* not in GGUF exports → MTP unavailable on GGUF path initially; record in README/status |
| MTP forward | new `qwen3_5_mtp.cpp` (mirror `qwen3_5_mtp.py`) reusing our decoder-layer building blocks in `qwen3_5.cpp` / `qwen3_5_dense.cpp` | fc-cat-norm head + one full-attn (+MoE) layer + final norm; bf16 GEMMs |
| hidden tap | `src/vllm/model_executor/models/qwen3_5.cpp:3074-3094` (post-final-norm `dnorm` + logits gather) | keep/expose the FULL `[T,H]` post-norm hidden for the drafter (today only gathered rows survive); the tap tensor is exactly upstream's `hidden_states` input |
| speculator | new `src/vllm/v1/worker/gpu/spec_decode/` (mirrored tree: `speculator.*`, `autoregressive/`, `mtp/`) | port `prepare_prefill_inputs` as a CUDA kernel (mirror the Triton at `autoregressive/speculator.py:469-588`); k=1 early-exit first |
| rejection sampler | `src/vllm/v1/sample/sampler.cpp` (greedy = `GreedyArgmaxHost` `:164`) + new `rejection_sampler.*` | greedy: argmax over the expanded `[Σ(1+k_i), vocab]` logits + equality (mirror `_rejection_kernel` greedy branch); keep on-device (our logits already stay device-side, `qwen3_5.h:72-79`) |
| input batch | `include/vllm/v1/worker/gpu/input_batch.h` | add `num_draft_tokens`, per-req draft counts, expanded `logits_indices`/`cu_num_logits` (mirror `input_batch.py:47-178`) |
| runner | `src/vllm/v1/worker/gpu/runner.cpp:704` (sampler call) | route through rejection sampler when drafts scheduled; call `propose()` post-sampling; return draft tokens out-of-band (`take_draft_token_ids` mirror) |
| scheduler | `include/vllm/v1/core/sched/scheduler.h:217-218` (`num_lookahead_tokens_ = 0`), `output.h:148` (`scheduled_spec_decode_tokens` always empty), `scheduler.cpp:398-408` (rollback DEFERRED comment), `request.h` (no `spec_token_ids` yet) | set lookahead=k for mtp; schedule `spec_token_ids` (mirror `scheduler.py:593-609`); implement rollback (mirror `:1580-1612`); add `update_draft_token_ids` (mirror `:1937`) + engine-core hookup (`core.py:515-517`); `allocate_slots` already takes `num_lookahead_tokens_` (`scheduler.cpp:163,234`) |
| GDN backend | `gdn_attn.{h,cpp}` + `cuda_gdn.cu` + conv kernel | §3.1 |
| KV manager | our mamba single-type manager | `num_speculative_blocks` (§2.6) |
| config | engine args / OpenAI server | `--speculative-config '{"method":"mtp","num_speculative_tokens":k}'` parsing; auto-resolve arch (mirror `speculative.py:480-489`) |

## 5. Phased plan with gates

Denominator discipline (from B5 + benchmark-protocol.md): once spec decode is
on, **the honest vLLM baseline is vLLM WITH the same speculative config**, at
the same operating point. Acceptance is content-dependent (B5: prose ~2.0/15
vs code ~5.5/15 accepted) — synthetic random-token gate prompts show ~nothing;
A/B must use natural-language/code workloads. Spec decode can go <1× at high
concurrency (bs-decay, B5: 4.0×→1.9× c1→c32 on a 9B) — gate at BOTH the
latency operating point (low concurrency, where vLLM's own defaults enable
spec) and the throughput gate concurrency, mirroring vLLM's behavior at each.

- **M-mtp-0 — weights + head parity (no engine changes). CLOSED 2026-07-24.**
  Load `mtp.*` on both models; run the MTP head standalone on captured target
  hidden states; compare against a vLLM-oracle capture of `Qwen3_5MTP` on the
  same inputs. Gate: argmax match on ≥16/16 positions across a natural-language
  prompt set, both checkpoints.
  **RESULT — op-level parity against a dumped oracle, NOT a token-generation
  SACRED gate.** Oracle: the installed vLLM 0.25.0 executable whose
  `qwen3_5_mtp.py` is byte-identical to the pin `e24d1b24` (md5
  `fbc42adca54090f7be1d685e1ee624cf` on the wheel, the pinned tree and
  `dgx:~/work/vllm-pin`), driven with
  `speculative_config={"method":"mtp","num_speculative_tokens":1}`,
  `enforce_eager=True`, one 27-token natural-language prompt; the harness
  intercepts the first real `Qwen3_5MTP.forward` and stores its exact draft
  inputs and outputs. Goldens
  `tests/parity/goldens/qwen3_5_mtp_head_{27b,35b}/`, consumed by the focused
  case `tests/parity/test_op_parity.cpp:1914` (runner `:1373`), 20/20
  assertions with `VLLM_MTP_REQUIRE_CHECKPOINTS=1`. Both checkpoints, 27 rows:
  - **argmax exact on 26/26 rows whose oracle top-1 is unambiguous.** The one
    remaining row per checkpoint (27B row 0, 35B row 7) is an **EXACT oracle
    top1==top2 tie** — 14.375 == 14.375 and 20.25 == 20.25 bit-for-bit in
    vLLM's own f32 logits, where vLLM's *own* `argmax` and `topk` disagree with
    each other. Our pick equals the oracle's `topk` top-1, i.e. a tied maximum,
    so that row carries no correctness information. Every other row has an
    oracle margin ≥ 0.125.
  - **logits within the whole-model bound** (atol 0.05 + rtol 0.05, the exact
    tolerance `qwen36_logits_{27,35}b` use): **0/216 out of tolerance on both**;
    max |Δ| 0.159 (27B, rel 1.5 %) and 0.125 (35B, rel 0.78 %) over vLLM's
    top-8 per row.
  - **shared lm_head isolated** (vLLM's own hidden states → our shared
    `lm_head`): **bit-exact on the 35B** (max |Δ| = 0, argmax 26/26) and
    max |Δ| 0.062 on the 27B (bf16 GEMM ordering; argmax 26/26). This proves
    the `load_eagle_model` embed/lm_head sharing and the NVFP4 logits GEMM
    independently of the head forward.
  - **hidden states** are bf16 on both sides, so an element-wise absolute bound
    below one ULP (0.0625 at |h| ≈ 8) is unsatisfiable even for a perfect port;
    the gate is therefore on direction and scale, which a structural error
    (wrong norm style, transposed `fc`, dropped gate, wrong rope) would move:
    min per-row cosine **0.99949** (27B) / **0.99863** (35B), global best-fit
    scale **0.99969** / **0.99996**, relative RMS **0.95 %** / **1.20 %**.
    Reported for the record, not gated: 370/138240 (27B) and 798/55296 (35B)
    elements exceed atol 0.05 + rtol 0.05, concentrated in the high-magnitude
    channels.
  - compute-sanitizer memcheck over both heads: **0 errors** (gate still 20/20
    under the sanitizer). `test_mtp_speculator` 7/7 cases / 141 assertions.
  - **bf16-unquant carve-out verified from the checkpoints**: 27B has exactly
    15 `mtp.*` tensors and the 35B exactly 19, **all BF16** in the safetensors
    headers. Correction to §1's premise: at these snapshots the quant configs
    DO exempt them (27B `compressed-tensors` exclude `re:^mtp\..*`; 35B
    `MIXED_PRECISION` exclude `mtp*`, `mtp.layers.0*`), so the upstream
    `mtp.fc` gotcha is not live here — and our loader is unconditional anyway:
    every `mtp.*` tensor goes through `LoadBf16Direct`/`LoadBf16RawNK`
    (`src/vllm/model_executor/models/qwen3_5_mtp.cpp:35,48`), which hard-fails
    on any non-BF16 dtype, so a quantized path cannot be taken silently.
  - **Harness note (upstream drift found and handled):** both gate checkpoints
    declare `*ForConditionalGeneration`, so vLLM 0.25.0's speculator takes its
    multimodal branch and calls the draft with `input_ids=None` plus a prebuilt
    `inputs_embeds` (`v1/spec_decode/llm_base_proposer.py:962-991`). The dump
    tool therefore also patches `Qwen3_5MTP.embed_input_ids` (called with the
    real ids one line earlier, `:971`) and recovers the token ids from there,
    refusing to write a golden unless the recovered ids cover the whole forward.
- **M-mtp-1 — 27B k=1 greedy e2e (includes the GDN spec path — see §0).**
  Scheduler plumbing + prepare-prefill kernel + greedy rejection + GDN slot
  mechanism (k+1=2 slots). **I2 (scheduler-half) is LANDED — see §2.7 for the
  frozen metadata ABI; the remaining I3/I5 work codes against it.** The row
  stays `GATING`: I2 is host-side plumbing that is INERT by default and buys no
  gate on its own; nothing is claimed until the e2e token gate below runs.
  Gate: token-for-token 16/16 vs
  `vllm --speculative-config '{"method":"mtp","num_speculative_tokens":1}'`
  greedy, AND acceptance-rate telemetry within noise of vLLM's on the same
  prompts. Then A/B decode throughput + TPOT (expected from B5/community:
  ~1.15–1.35× decode at conc16/32; +27.5% k=1 reported on the 35B-class).

  **M-mtp-1 SUB-INCREMENT DECOMPOSITION (scoping pass 2026-07-24).** M-mtp-1 is
  NOT one landing — I2 (scheduler-half), I3 (verify/rejection), I4 (GDN spec
  slots) already landed against the §2.7 frozen ABI; the remaining loop is four
  more independently-gateable bricks. The e2e token gate above lands with the
  LAST (I5d). Order and boundaries:
  - **I5a — GDN layer spec routing + runner spec-metadata upload. LANDED
    2026-07-24 (`CLAIM-SPEC-MTP-I5A`).** The `GdnBlockPaged` `num_spec_decodes>0`
    branch (PURE-spec fast path) routing a spec batch through
    `vt::CausalConv1dSpecUpdate` + `vt::GdnSpecDecode` (mirror
    `qwen_gdn_linear_attn.py:1344-1357,1455-1475`), plus the `StepDevInputs` /
    `BuildStepDevInputs` upload of I4's six spec device tensors
    (`spec_state_indices_tensor`, `spec_query_start_loc`, `spec_token_indx`,
    `non_spec_token_indx`, `spec_sequence_masks`, `num_accepted_tokens`) and the
    decode-graph `Refresh` copies, and the `ValidateGdnAttentionMetadata` spec
    contract. DEFAULT-OFF INERT (`num_spec_decodes==0` ⇒ stub uploads, non-spec
    branch, byte-identical). Gate: a synthetic test drives the spec branch at the
    real 27B/35B GDN dims (Hv 48/32, Dk=Dv 128) and asserts BIT-EXACT vs the I4
    ops applied as a token-sequential decode chain (`GdnBlockPagedForTest`);
    RED-first proven by a reverted stub (spec recurrence zeroed ⇒ 4/8 fail,
    maxΔ 1.3-1.6). NOT the e2e gate — the MIXED spec+non-spec batch (index_select
    split + merge) is refused loudly and lands with I5d's runner loop. The runner
    `GDNAttentionMetadataBuilder::build` spec feed (`num_decode_draft_tokens_cpu`,
    `num_accepted_tokens`) is I5d (the runner loop), inert until then.
  - **I5b — `prepare_prefill_inputs`. LANDED 2026-07-24 (`CLAIM-SPEC-MTP-I5B`).**
    Ports the draft-prefill input shift (`autoregressive/speculator.py:469-588`):
    per request shift `input_ids` left one within its query span, splice the
    just-sampled next token (`num_sampled>0 ? last_sampled[idx_mapping[r]] :
    next_prefill_tokens[...]`) into the freed last slot, `query_len -=
    num_rejected`, emit `last_token_indices`/`query_start_loc`/`seq_lens` +
    request-count CUDA-graph padding, into the `SpecPrefillInputs` output struct
    (`include/vllm/v1/worker/gpu/spec_decode/autoregressive/prepare_prefill_inputs.h`,
    `src/.../prepare_prefill_inputs.cpp`). KEY INVARIANT: input_ids/positions keep
    the TARGET batch's total size (rejected positions PADDED not compacted, NOTE
    :162-167) so the draft reuses the target attention metadata; only the shift
    range / last_token_index / sampled row shrink by num_rejected. **Implemented as
    a HOST routine** (no new CUDA kernel): OUR `prepare_inputs`/
    `combine_sampled_and_draft_tokens` are host `std::vector` routines marked
    DEVICE-NEUTRAL (the DGX leaf ports the loop to the Triton kernel over the draft
    `input_buffers`), and this is exactly that family — so the "CUDA==CPU"/
    compute-sanitizer clauses are N/A and no dgx run is needed. Unit-gated against
    a from-the-algorithm oracle: `tests/vllm/v1/spec_decode/test_prepare_prefill_inputs.cpp`
    7 cases / 27 assertions (perfect-accept, early-mismatch k=1 j=0 + k=3 j=1,
    single-token, multi-request, chunked-prefill, CG padding), RED-first by a
    reverted no-shift/no-splice stub (7/7 fail, 12/27 assertions). DEFAULT-OFF
    INERT (nothing calls it until I5d); additive by construction (two new files +
    one new test + 2 CMake lines, no existing TU touched) → all SACRED gates
    byte-identical, no re-run owed. `SPEC-MTP` STAYS `GATING`.
  - **I5c — MTP paged propose forward + draft KV layer. LANDED 2026-07-24
    (`CLAIM-SPEC-MTP-I5C`).** `Qwen3_5MTPModel::ForwardPaged` runs the fc-cat-norm
    head + one `layer_type="full_attention"` decoder layer over the head's OWN
    paged draft KV layer via `FullAttnBlockPaged` (ReshapeAndCache + PagedAttention
    over the target's slot_mapping/block_table) — mirror of `qwen3_5_mtp.py:129-165`
    over the paged backend (`speculator.py:346` `_run_model`). The draft KV layer is
    added by `MakeQwen3_5KVCacheSpec(..., num_spec>0)` as an extra `FullAttentionSpec`
    group (`fa_draft`, index `num_hidden_layers`, `speculator.py:163-169`), sized like
    a target full-attn layer; `num_spec==0` (production default) is the two pre-I5c
    groups byte for byte. The target forward exposes the `[T,H]` post-final-norm
    hidden via `ForwardDeviceTap` (an INERT optional copy of `dnorm` before the logits
    gather; nullptr on every shipped caller). `MtpProposePrefill`
    (`src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp`) is the callable k=1
    propose: I5b `prepare_prefill_inputs` → one `ForwardPaged` → argmax at
    `last_token_indices` (`speculator.py:236-238` early-exit). **NOT wired into the
    runner step loop (I5d).** Gates: paged-vs-standalone parity on BOTH checkpoints
    (`test_op_parity` RunQwen35MtpHead extended); a synthetic two-step draft-KV
    write/read (bit-exact vs the combined forward, RED control = fresh-KV diverges);
    draft-KV sizing (`test_model_registry`); `MtpProposePrefill` k=1 — 5 new
    `test_mtp_speculator.cpp` cases. DEFAULT-OFF INERT; the shared-`qwen3_5.cpp`
    SACRED gates (27B/35B/Coder) re-run and pass byte-identically. `SPEC-MTP` STAYS
    `GATING` (the runner loop + e2e token gate are I5d).
  - **I5d-pre — registry/forward-seam enabling refactor. LANDED 2026-07-25
    (`CLAIM-SPEC-MTP-I5D-PRE`).** A scoping pass found the model seam is fully
    TYPE-ERASED, so the runner cannot reach the concrete target weights, the
    hidden-state tap, or the loaded MTP weights the I5d verify/propose loop needs.
    This increment adds those access paths as ADDITIVE, INERT-WHEN-SPEC-OFF changes
    (no spec loop — that is I5d), plus one latent-bug fix. FOUR seam changes:
    (1) **hidden-tap out-field** on the type-erased `ModelForwardInput`
    (`include/vllm/model_executor/models/model_registry.h:106-133`): a
    `Qwen3_5MTPHiddenStates* hidden_tap = nullptr`; the Qwen3.5 dense + MoE
    `ModelForwardFn` (`qwen3_5_dense.cpp` / `qwen3_5_moe.cpp` `ForwardQwen3_5{Dense,Moe}`)
    route to the EXISTING `Qwen3_5{,Dense}Model::ForwardDeviceTap` when it is
    non-null, else the current path — byte-identical when null (every spec-off run);
    other models ignore the field. (2) **draft-construction access on `LoadedModel`**
    (`model_registry.h:95-131` + `model_registry.cpp`): a virtual `BuildMtpDraft(config)`
    that constructs the `Qwen3_5MTPModel` draft from the retained MTP weights + this
    model's own target weights (only the concrete `Qwen3_5{Dense,Moe}LoadedModel` can;
    default returns null for non-MTP), plus `AttachMtpDraftWeights` to retain them and
    `supports_mtp_draft()`. (3) **MTP weight loading + shard retention in
    `FromModelDir`** (`src/vllm/entrypoints/model_loader.cpp`): when
    `EngineParams::speculative_config` is an MTP config, `FromModelDir` calls
    `LoadQwen3_5MTP(*shards, config, kind)` while the local `shards` shared_ptr is
    still alive (the dense direct-device-load path releases shards post-load) and
    attaches them to the model; GGUF + MTP is rejected (no `mtp.*` in GGUF); spec
    unset ⇒ load path unchanged. (4) **runner ctor widening + the LATENT
    full-attn-group bug** (`include/vllm/v1/worker/gpu/runner.h`,
    `src/vllm/v1/worker/gpu/runner.cpp`): the `GPUModelRunner` ctor is widened with
    optional draft model + draft KV cache + `SpeculativeConfig` (defaulted/nullopt →
    today's behavior at every existing construction site + runner test). **LATENT-BUG
    FIX:** `initialize_kv_cache` assigned `full_attn_group_id_` to the LAST
    `kFullAttention` group; with a third `fa_draft` group (`num_spec>0`) it would
    wrongly select `fa_draft`. Now it selects the TARGET group deterministically — the
    FIRST (index 0) non-eagle full-attn/MLA group — so a later-appended draft group
    can never displace it. With `num_spec==0` (every run today) there is exactly one
    full-attn group, so the fix is BYTE-IDENTICAL now (proven RED-first: the buggy
    last-wins loop reports group 2 with a synthetic third `fa_draft` group, the fix
    reports 0). Unit coverage (RED-first): the hidden-tap out-field captures the
    post-final-norm `[T,H]` and is byte-identical on logits
    (`test_qwen27_paged_forward.cpp`); `BuildMtpDraft` returns a valid draft for a
    Qwen3.5 target and null for a non-MTP model
    (`test_mtp_speculator.cpp`); the `full_attn_group_id_` selection picks the target
    group with a synthetic third `fa_draft` group present, unmarked and eagle-marked
    (`test_runner.cpp`). DEFAULT-OFF INERT throughout; `SPEC-MTP` STAYS `GATING` — this
    is the enabling seam, NOT the spec loop and NOT any e2e result. `benchmark_binding=false`,
    no speed claim.
  - **I5d — config plumbing + runner loop. PARTIAL, 2026-07-25 (`CLAIM-SPEC-MTP-I5D`).**
    LANDED and spec-OFF byte-identical: `--speculative-config` JSON parse
    (`ParseSpeculativeConfigJson` -> `EngineParams::speculative_config`), the
    `LoadedEngine` resolution (`ResolveSpecConfig` -> `ResolveMtp`, the widened KV
    spec via `MakeQwen3_5KVCacheSpec(num_spec>0)`, `BuildMtpDraft`, forced sync
    scheduling, `MakeScheduler(spec)`, `EngineCore(check_for_draft_tokens=true)`),
    and the full runner verify/propose loop: draft splice
    (`update_req_spec_token_ids`), hidden-tap capture, the GDN builder spec-overload
    feed (`num_accepted_tokens` + `num_decode_draft_tokens` -1 sentinel), the GDN
    compact-pool k+1 state-slot remap + widened conv cache + draft-KV allocation,
    `MtpProposePrefill` post-sampling, `take_draft_token_ids` out-of-band, and
    acceptance telemetry. Builds CLEAN on dgx (CUDA `-Werror` 0 warnings, cutlass-ON
    banner). **Spec-OFF SACRED regressions PASS (byte-identical, all changes gated
    on `spec_on()`): 27B 235/235, 35B 315/315, Coder 138/138; unit test_runner
    257, test_mtp_speculator 169, test_gdn_metadata_builder 483, test_ops_gdn 3630.**
    **The three-way 27B token gate is NOT yet passing** — `SPEC-MTP` STAYS `GATING`.
    RCA (measured, `tests/parity/test_qwen27_spec_decode.cpp`): the spec-ON engine
    resolves config, allocates the widened KV + draft KV, runs the target forward
    with the hidden tap, and reaches the GDN layer, then THROWS on the FIRST
    (prefill) step at `vt: gdn_state_gather: working/cache row shapes must match`
    (`src/vt/ops.cpp:1773`). The gap: I4's spec conv rollback REQUIRES the conv
    state row widened to `(conv_kernel-1)+num_spec` (`mamba_utils.py:226`), but the
    NON-spec GDN conv ops (`GdnStateGather`/`CausalConv1dFwd`/`CausalConv1dUpdate`,
    `GdnBlockPaged` prefill/decode branches) assume a `(conv_kernel-1)`-wide row, so
    the prefill conv gather shape-mismatches the widened cache. Closing the gate
    needs the non-spec GDN conv ops made widened-cache-aware (mirror vLLM
    `causal_conv1d` with `state_len = width-1 + (seqlen-1)`,
    `qwen_gdn_linear_attn.py:1181-1184`), a kernel-level change with its own
    bit-exact gate — plus the MIXED spec+non-spec `GdnBlockPaged` split/merge (still
    refused loudly; only trips under concurrency, so a single-request gate would not
    need it). NOT overclaimed: token-identity + nonzero acceptance is the bar and it
    is not yet met. The remaining code (config + loop) is a real, gated,
    regression-green partial. `benchmark_binding=false`, no speed claim.
  - **I5d-followup (the OPEN e2e gate) — config plumbing + runner loop DONE; still
    owed for `SPEC-MTP` to leave `GATING`:** (1) widened-cache-aware non-spec GDN
    conv ops (the measured blocker above), (2) the MIXED `GdnBlockPaged` split/merge
    (needs a row `IndexSelect`/`IndexCopy` vt op), then (3) the three-way token gate
    (our-ON == our-OFF == vLLM `--speculative-config mtp` greedy, token-for-token)
    with measured nonzero acceptance, then (4) A/B throughput vs vLLM same-config.
  - **I5e — widened-cache-aware non-spec GDN conv ops + the PASSING three-way 27B
    gate. LANDED 2026-07-25 (`CLAIM-SPEC-MTP-I5E`). `SPEC-MTP` LEAVES `GATING`.**
    Closes the I5d blocker (1) AND drives the gate to a MEASURED PASS on single-request
    greedy. TWO code changes, both byte-identical at `num_spec==0`:
    - **Widened-cache-aware non-spec GDN conv ops.** The spec conv cache row is
      widened to `(K-1)+num_spec` taps (`mamba_utils.py:226`), but the non-spec ops
      assumed a `(K-1)`-wide row and shape-mismatched the widened cache (the I5d throw
      at `src/vt/ops.cpp:1773`). Fix — mirror vLLM's non-spec `state_len=KERNEL_WIDTH-1`
      + PHYSICAL `stride_conv_state_tok` (`causal_conv1d.py:66-69,156`; prefill stores
      the leading `[0,K-1)` cols at `:233`): `CheckConvCommon`/`CheckGdnStateIo` admit a
      cache inner-dim ≥ working inner-dim; `GdnStateGather`/`GdnStateScatter` (CPU+CUDA)
      copy the LEADING `work_inner` cols per channel with the cache row's physical
      stride (contiguous fast path kept when `cache_inner==work_inner`, so the SSM
      rank-4 cache and every non-spec model are bit-identical); `CausalConv1dUpdate`
      (CPU + scalar/fast CUDA) addresses the row by physical `state_len=conv_state.shape[2]`
      while operating on the leading `(K-1)` taps. `CausalConv1dFwd` needs NO change —
      in `GdnBlockPaged` it only ever runs on the GATHERED narrow working buffer, so the
      gather does the sub-window extraction. Files: `src/vt/ops.cpp`, `src/vt/cpu/cpu_ops.cpp`,
      `src/vt/cuda/cuda_gdn.cu`. Op-gate: `test_ops_gdn` grows a widened-cache
      byte-identity case (leading window == narrow cache after update + gather/scatter,
      extra tap column inert) — 3678/3678 (RED-first: the throw was the real I5d gate).
    - **Async input-combine forced off under spec (the 0-acceptance RCA).** With the
      conv seam fixed the loop RAN end-to-end and three-way token identity PASSED, but
      MEASURED acceptance was 0/31 — a dead drafter. RCA (systematic-debugging, env-gated
      instrumentation on the real 27B): the drafts were CORRECT (matched the target's next
      argmax) but the runner's async input-combine (`async_input_combine_`, default ON —
      `AsyncRunnerFlagIsOn(nullptr)==true`) splices the device-resident `last_sampled`
      token over each decode row with `num_new_sampled_tokens==1`; it is NOT spec-aware
      and overwrote the DRAFT position of the verify batch with the committed token, so
      `draft_sampled[1]` read the committed token and the rejection sampler rejected every
      draft. Fix — force `async_input_combine_ = AsyncRunnerEnvDefault() && !spec_config_.has_value()`
      in both `GPUModelRunner` ctors (spec already forces SYNC scheduling and gets its
      drafts spliced into `token_ids_cpu` by `update_req_spec_token_ids`+`prepare_inputs`;
      byte-identical for non-spec, `spec_config_` nullopt). File: `src/vllm/v1/worker/gpu/runner.cpp`.
    - **THE GATE — MEASURED PASS** (`tests/parity/test_qwen27_spec_decode.cpp`, single-request
      greedy, 27B `~/bench/q36-27b-nvfp4-vllm`): three-way token identity PASS
      (our-spec-ON == `greedy_ids.npy` = the vLLM `--speculative-config mtp` greedy ==
      our-spec-OFF continuation, token-for-token on the 16-token golden prefix;
      continuation " capital of Germany is Berlin."), **acceptance 16/16 drafts accepted**
      (100% on this deterministic factual prompt; ~16 target decode steps saved over the
      32-token run), 32 tokens produced. `proposed>0 && accepted>0` both PASS.
    - **Regressions (all on the FINAL code, clean cutlass-ON build).** Spec-OFF SACRED
      byte-identical, STANDALONE under `flock $HOME/gpu.lock`, one big-model at a time:
      27B 235/235, 35B 315/315, Coder 138/138. `test_ops_gdn` 3678/3678,
      `test_gdn_metadata_builder` 483/483. compute-sanitizer 0 on the now-reachable spec
      step. `SPEC-MTP` moves past `GATING` (single-request greedy correctness PROVEN);
      NOT `DONE` — the MIXED `GdnBlockPaged` split/merge (concurrency) + the throughput
      A/B vs vLLM same-config are **I6**. `benchmark_binding=false`, no speed claim.
  - **I6 — the c1 throughput A/B vs vLLM same-config (THE §5 THROUGHPUT GATE, low-
    concurrency operating point). LANDED 2026-07-25 (`CLAIM-SPEC-MTP-I6`).
    `benchmark_binding=true`** — the first spec-decode speed number. Measures the §5
    gate at c1 (the operating point the engine supports; the c>1 mixed-batch path is
    still refused). Both arms spec-ON, SAME `{"method":"mtp","num_speculative_tokens":1}`,
    27B `~/bench/q36-27b-nvfp4-vllm`, greedy, c1, 8 real prompts x 256 out, prose + code
    (random tokens accept ~nothing), idle box, one engine at a time under one
    `flock`, 3 reps (cold TTFT leg discarded). OURS = `examples/vllm-bench` + an
    additive `--speculative-config` flag (production config). vLLM = pinned 0.25.0
    `vllm serve --speculative-config ... --no-enable-prefix-caching` (graphed
    PRODUCTION: `enforce_eager=False`, `FULL_AND_PIECEWISE`, inductor) + `vllm bench
    serve --dataset-name sharegpt --sharegpt-output-len 256 --ignore-eos
    --max-concurrency 1`; vLLM confirmed engaging MTP (`Resolved architecture:
    Qwen3_5MTP`, live `SpecDecoding metrics`). Correctness re-confirmed on the clean
    build FIRST (`test_qwen27_spec_decode` PASS, 16/16 accepted), so the timings are
    from a token-identical loop.
    - **RESULT — OURS spec-ON AT/ABOVE vLLM spec-ON on EVERY measured axis at c1**
      (median tok metrics, prose / code): **TPOT** ours 66.2 / 62.95 ms vs vLLM
      69.1 / 65.3 ms (**ours ~1.04x faster**); **output tput** ours 15.10 / 15.72
      vs vLLM 14.43 / 15.13 tok/s (**+4.6% / +3.9%**); **ITL** ours 121.6 / 121.1 vs
      vLLM 123.2 / 123.2 ms (ours marginally lower); **TTFT** (warm) ours 131.3 /
      131.0 vs vLLM 151.5 / 181.2 ms (ours lower, vLLM TTFT noisy); **acceptance**
      ours 0.85 / 0.92 vs vLLM 0.838 overall (per-position 0.81-0.84) — ours >=,
      within noise, live drafter both sides; **peak host RSS** ours 28.4 GB spec-ON
      (24.8 GB spec-OFF, the +3.6 GB = draft KV + widened conv + draft head, spec
      3.1 GDN doubling), vLLM reserves 24.97 GiB weights + 26.2 GiB KV at util 0.45 —
      both well inside the 119 GiB pool.
    - **Baseline (does spec help?)** — YES on both sides at c1: ours TPOT
      100.5->66.2 prose (1.52x) / 100.0->62.95 code (1.59x); vLLM 104.35->69.1
      (1.51x) / 104.35->65.3 (1.60x). At spec-OFF ours is already ~4% faster than
      vLLM and preserves the lead spec-ON. **Noise:** ours reruns TPOT s<0.5%, vLLM
      ~2%; the ~4% margin exceeds the band (real).
    - **Disposition:** the §5 THROUGHPUT gate is MET at c1 (on-par-or-above vLLM on
      every axis, token-identity proven). `SPEC-MTP` stays `ACTIVE` (not `DONE`)
      because the project's every-axis rule also spans the higher-concurrency
      operating point, and spec at c>1 is unimplemented (mixed spec+non-spec
      `GdnBlockPaged` split/merge still refused). Remaining for `DONE`: (1) the mixed
      `GdnBlockPaged` split/merge (needs a row `IndexSelect`/`IndexCopy` vt op) + a
      c>1 A/B, (2) a user-facing supported `--speculative-config` on the OpenAI
      server (the bench flag is example-only/additive). Raw logs on dgx
      `~/work/mtp-bench-i6/{results,vresults}`.
  - **I7 — the MIXED spec+non-spec GDN batch (concurrency) + server/CLI
    `--speculative-config` + the c>1 A/B. LANDED 2026-07-25 (`CLAIM-SPEC-MTP-I7`).**
    Closes the two `DONE`-blockers from I6. THREE pieces:
    - **New row gather/scatter op `vt::IndexSelect`/`vt::IndexCopy`** (`kIndexSelect`/
      `kIndexCopy`; CPU `src/vt/cpu/cpu_ops.cpp` + CUDA `src/vt/cuda/cuda_gdn.cu`,
      dispatch/validation `src/vt/ops.cpp`, decl `include/vt/ops.h`). torch
      `index_select(0,·)` / `index_copy_(0,·)` over rows: dtype-agnostic byte copy,
      the base side may carry an outer (padded / merged-view) row stride, inner
      contiguous. Gate: CUDA==CPU bit-exact at real GDN row widths (conv_dim 10240,
      dcore 6144), bf16 + f32, incl. a row-strided source and a full-partition
      scatter round-trip (`tests/vt/test_ops_gdn.cpp`); RED-first proven by a
      neutered gather (524/526 fail).
    - **Mixed `GdnBlockPaged` split/merge** (`GdnBlockPagedMixedSpec`,
      `src/vllm/model_executor/models/qwen3_5.cpp`) — mirror of
      `qwen_gdn_linear_attn.py::_forward_core:1329-1576`: index_select `mixed_qkv`/
      `a`/`b` by `spec_token_indx`/`non_spec_token_indx` into compact per-group
      buffers, run the I5a spec conv+recurrence over the spec rows + the widened-
      cache-aware prefill conv+recurrence over the reclassified non-spec rows
      (upstream forces `num_decodes==0` when a spec row exists, gdn_attn.py:243-251),
      merge the two core outputs back with index_copy. Reuses the shipped spec/
      prefill vt ops verbatim (each already bit-exact-gated). One validation fix:
      the non-spec cu_seqlens spans the non-spec tokens (`nd_tok+np_tok`), not
      `num_actual_tokens` — equal only at `num_spec==0`, so byte-identical on the
      default path. **MODEL-INDEPENDENT BIT-EXACT PROOF:** because the spec and
      prefill requests use DISJOINT state slots, the mixed batch's per-row output
      must equal the spec request run PURE-spec (rows 0..k) followed by the prefill
      request run PURE-prefill (`test_qwen3_5_gdn_spec_routing` "MIXED == pure spec +
      prefill", CPU bit-exact 27B+35B dims, CUDA within the M-retile band; RED-first
      by a broken merge). compute-sanitizer 0 on the mixed step + the new op.
    - **Server + CLI `--speculative-config`.** The server flag was wired in I5d
      (`examples/server/main.cpp`); I7 adds the CLI path — an ABI v6
      `vllm_model_params.speculative_config` field (`include/vllm.h`) parsed by
      `ParseSpeculativeConfigJson` in `vllm_engine_load` (`src/capi/vllm_c.cpp`) and
      the `--speculative-config` flag on `examples/cli/main.cpp`; absent => NULL =>
      byte-identical no-spec engine; malformed/unsupported => loud load failure.
      README CLI-flags table + spec-decode note updated.
    - **c>1 CORRECTNESS — the batch-nondeterminism finding (honest, decisive).**
      The 27B greedy output is bf16-BATCH-NONDETERMINISTIC at near-ties: spec-OFF at
      max_num_seqs=4 vs =1 differs on 2/3 short prompts, NO spec decode involved
      (`test_qwen27_spec_decode_concurrent` control). Because spec's verify step is a
      THIRD batch shape (1+k tokens/req), per-request EXACT spec-ON==spec-OFF at c>1
      is UNATTAINABLE for near-tie requests — the MODEL, not the mixed code (which is
      bit-exact-proven above). So the c>1 correctness bar is: (a) the mixed path runs
      and accepts drafts (integration: mixed-batch invocations>0, 43/51 accepted at
      the staggered c4 gate), (b) the model-independent bit-exact proof, and (c) the
      EXACT three-way identity still holds at c1 (I6). This mirrors
      near-tie-distributional-gate: token-exact where the model is deterministic (c1),
      distributional where it is not (c>1).
    - **c>1 THROUGHPUT A/B — ours vs vLLM, both spec-ON, same `--speculative-config`,
      27B, greedy, 8 prose+code prompts x 256 out, idle box, one engine at a time
      under one `flock`, 3 reps (rep1 discarded as cold), production configs.** OURS =
      `examples/vllm-bench` + `--speculative-config`; vLLM = pinned 0.25.0 graphed
      `vllm serve --speculative-config ... --max-num-seqs 8 --no-enable-prefix-caching`
      + `vllm bench serve --max-concurrency {2,4,8}`.
      **RESULT — spec decode KEEPS HELPING at every c>1 on BOTH engines; ours tracks
      vLLM (median output tput tok/s, prose / code):**
      - **c2** ours ON 28.9 / 29.8 vs OFF 19.0 / 19.0 (**1.52x / 1.57x**); vLLM
        28.4 / 29.1 tok/s (ours +1.6% / +2.5%).
      - **c4** ours ON 53.9 / 57.2 vs OFF 36.7 / 36.7 (**1.47x / 1.56x**); vLLM
        53.4 / 56.3 (ours +0.9% / +1.7%).
      - **c8** ours ON 100.5 / 104.1 vs OFF 70.3 / 70.3 (**1.43x / 1.48x**); vLLM
        99.7 / 105.3 (ours +0.9% / -1.1%, within vLLM's own run noise).
      TPOT spec-ON stays ~67-77 ms across c2-c8 (vs ~105-113 ms spec-OFF). Ours does
      NOT go neutral at c8 (the mixed batch keeps every request drafting).
      **Disposition: ours is ON-PAR-OR-ABOVE vLLM spec-ON at every measured c>1 axis** (output tput within ~+/-2%, ours slightly ahead at c2/c4, within-noise at c8; acceptance on-par 0.84-0.92 vs 0.835), tracking vLLM's ~1.5x spec speedup at every concurrency. The implementation is COMPLETE and at vLLM parity (mixed batch + bit-exact proof, server/CLI flag, c1 win I6). `SPEC-MTP` STAYS `ACTIVE`, NOT flipped to `DONE`, for one honest reason: the DONE criterion's strict `token-exact at c>1` clause is a proven MODEL impossibility — the 27B greedy is bf16-batch-nondeterministic (spec-OFF max_seqs 4-vs-1 differs 2/3 short prompts, no spec involved, affecting vLLM identically), so exact c>1 token identity cannot be met by any correct implementation; c>1 correctness is instead the model-independent bit-exact split/merge proof + acceptance parity (near-tie-distributional-gate), token-exact strict at c1. No lag, no missing work, no lever — the DONE final call is deferred to the user given this criterion ambiguity. Raw logs dgx `~/work/mixed-batch/{cN_results,cN_vresults}`.
- **M-mtp-2 — 35B k=1 greedy. CLOSED 2026-07-26 (`CLAIM-SPEC-MTP-M-MTP-2`).**
  Adds only the MoE MTP layer (reuses our MoE blocks) — GDN path identical.
  Same gates. Caveat: verify unions experts across k+1 tokens/request (more
  experts touched per step — measure, don't assume the 27B speedup transfers).
  **RESULT — the full 35B MoE-MTP e2e three-way token gate PASSES; the 27B
  speedup TRANSFERS to the MoE.** New gate `tests/parity/test_qwen36_spec_decode.cpp`
  (mirror of the 27B `test_qwen27_spec_decode.cpp` at 35B, MoE draft head
  `Qwen3_5MTPKind::kMoe` via `model_loader.cpp:582`) drives the pinned M0-exit
  prompt through the full paged engine with `speculative_config
  {"method":"mtp","num_speculative_tokens":1}`. RAN on dgx.casa (GB10, cutlass
  NVFP4 + FA2 ON), 9/9 assertions:
  - **Three-way token identity, STRICT (deterministic at c1):** our spec-ON ==
    `qwen36_logits_35b/greedy_ids.npy` (16/16), which is BOTH our spec-OFF
    (`test_qwen36_paged_engine.cpp`) AND vLLM's greedy. **Confirmed DIRECTLY
    against the live vLLM 0.25.0 oracle** (`tools/parity/capture_qwen36_spec_greedy.py`,
    `enforce_eager`, `gpu_memory_utilization=0.45`): vLLM spec-ON greedy AND
    vLLM spec-OFF greedy BOTH == the anchor, so all four decode paths are
    token-identical on the 16-token prefix.
  - **Acceptance (dead-drafter trap closed):** our 16/16 drafts accepted; vLLM's
    own acceptance 16/16 (rate 1.0) on the same prompt — parity, and the MoE
    drafter is provably alive.
  - **c1 SPEED A/B (our engine, spec-ON vs spec-OFF, same binary, 6 prose+code
    prompts × 128 out, greedy, reproducible across cold/warm σ<1%):** TPOT 11.80
    vs 14.03 ms (**1.19x faster**), output-tput 78.73 vs 67.68 tok/s (**+16.3%**),
    E2EL 1569 vs 1820 ms, draft acceptance **0.908** (364/401, in the 27B
    0.85-0.92 band). The MoE expert-union caveat does NOT erase the speedup.
  - **spec-OFF byte-identity:** the whole M-mtp-2 landing is a test + capture
    script + records diff (no source/kernel touched), so the 35B non-spec path
    is byte-identical and 35B SACRED 315/315 holds by construction
    (`git diff --stat` = test/docs only). CUDA `-Werror` 0 warnings (test + bench
    targets); no kernel touched ⇒ compute-sanitizer / device-leakage unchanged.
  Raw logs dgx `~/mtp35b/vllm.cpp/{gate.log,capON.log,capOFF.log,abON_warm.log,abOFF_warm.log}`.
- **M-mtp-3 — k>1 + stochastic rejection.** Multi-step decode loop
  (`_multi_step_decode` port; note upstream's own warning that k>1 re-runs the
  SAME single MTP layer and lowers acceptance, `speculative.py:805-813`),
  Gumbel draft sampling with position+1 noise alignment, full rejection
  sampler (non-greedy). Gate: distribution-level parity (greedy still 16/16;
  seeded-sampling trace match vs oracle where vLLM is deterministic), perf ≥
  vLLM same-config on every axis.
- **M-mtp-4 — CUDA-graph integration.** Mirror
  `autoregressive/cudagraph_utils.py` (prefill speculator graphs; FULL-decode
  graphs for k>1 steps; GDN spec metadata is capture-safe by the request-count
  padding at `gdn_attn.py:413-462`). Gate: no regression vs eager-spec; graphs
  on = our production config, same as the MVP gates.

## 6. Tests to port (protocol: .agents/test-porting.md)

Upstream `tests/` @ pin → our tiers:

| upstream | what it asserts | tier → ours |
|---|---|---|
| `tests/v1/spec_decode/test_mtp.py` (`test_mtp_load_model_unified` :67, `test_mtp_propose` :118, k=1) | MTP loader shares embed/lm_head; propose() shapes + k=1 flow | T-unit → `tests/vllm/v1/spec_decode/test_mtp_speculator.cpp` |
| `tests/v1/worker/test_gpu_autoregressive_speculator.py` (:52,:69) | `_run_model` tuple-vs-tensor return contract for MTP heads | T-unit (same file) |
| `tests/v1/spec_decode/test_rejection_sampler_utils.py` (`test_greedy_rejection_sample` :183 k∈{1,3}, `test_stochastic_rejection_sample` :141, `test_placeholder_draft_token_rejected` :285, block-verification :325,:372) | greedy accept-iff-argmax-match incl. bonus token; stochastic distribution; edge cases | T-unit + T-parity (golden logits dumps) → `tests/vllm/v1/spec_decode/test_rejection_sampler.cpp`; stochastic/block cases SKIPPED until M-mtp-3 (rule 6) |
| `tests/v1/sample/test_rejection_sampler.py` (`test_perfect_match` :133, `test_early_mismatch` :154, `test_multiple_sequences` :179, `test_single_token_sequence` :204, `test_empty_sequence` :225, `test_multiple_mismatches` :246, parametrized :288) | end-to-end sampler-level accept/reject token streams (legacy-sampler suite — port the ASSERTIONS, realized against our sampler) | T-unit (same file as above) |
| `tests/v1/worker/test_gpu_rejection_sampler_i64.py` (:109) | >2³¹ logits-buffer indexing | T-unit, SKIPPED until block verification exists |
| `tests/v1/attention/test_gdn_metadata_builder.py` (`test_gdn_build_classification` :175 — `mixed_decode_and_spec_decode` (the upstream #34845 crash), `pure_spec_decode`; `test_has_initial_state_after_reclassification` :187; `test_full_cudagraph_spec_metadata_uses_request_count` :201) | spec/non-spec split + decode→prefill reclassification + CG padding | T-unit → EXTEND our existing `tests/vllm/v1/attention/test_gdn_metadata_builder.cpp` (today asserts `num_spec_decodes == 0` at :74,:112 — those flips to real cases) |
| GDN spec kernel (no dedicated upstream unit test — kernel is covered via e2e; `tests/kernels/mamba/cpu/test_cpu_gdn_ops.py` has the CPU reference of `fused_sigmoid_gating`) | slot-select initial state + per-position snapshot bit-exactness | T-parity → extend `tests/vt/test_ops_gdn.cpp` + `tools/parity/` golden dumps vs the fla Triton kernel run in the oracle venv |
| `tests/v1/e2e/spec_decode/test_spec_decode.py::test_mtp_correctness` (:790; the `qwen3_5-hybrid` param :776-786 is our exact architecture) | spec-on vs spec-off output match (≥6/10 exact) + GSM8k accuracy floor | T-e2e → our paged-engine parity tests (`tests/parity/test_qwen{27,36}_paged_engine.cpp`) grow a spec-decode config; we gate STRICTER (16/16 greedy, per gates.md) |
| `tests/v1/spec_decode/test_acceptance_length.py` (:221-231 pattern) | acceptance-length telemetry against reference values | T-e2e (nightly dgx): acceptance-rate assertion vs vLLM-oracle-measured reference on the same prompt set |
| `tests/v1/core/test_scheduler.py` spec-decode cases (`test_schedule_spec_decode`, rollback cases) | scheduler spec-token scheduling + `num_computed_tokens` rollback | T-unit → extend `tests/vllm/v1/test_scheduler.cpp` (mapping per test-porting rule 3) |

DoD includes all non-SKIPPED tiers green (CI CPU tiers + nightly dgx GPU
tiers).

## 7. Effort ranking (impl order within M-mtp-1)

1. Scheduler + request plumbing (spec_token_ids, lookahead, rollback) — pure
   host C++, fully unit-testable, no kernels. **S**
2. Loader + MTP forward (27B) — bf16 GEMM reuse. **S/M**
3. Greedy rejection + input-batch expansion — one small kernel + host glue. **S**
4. `prepare_prefill_inputs` kernel port. **S**
5. GDN spec path (metadata split + kernel T>1/IS_SPEC + conv update +
   allocation) — the largest single piece; everything is a 1:1 port of cited
   lines. **M/L**
6. 35B MoE MTP layer. **S** (reuses MoE blocks)
7. k>1 + stochastic + CUDA graphs. **M**

## 8. Open questions (tracked, none blocking M-mtp-1)

- `mamba_cache_mode` interplay: upstream Qwen3_5MTP rejects `"all"` prefix
  caching (`qwen3_5_mtp.py:206-210`) and the align-mode spec-copy has a conv
  layout carve-out (`mamba_hybrid.py:136-142`) — mirror the rejection, check
  which mode our engine runs (we mirror non-align V0 semantics today).
- Draft-side CUDA-graph shape for k=1 (prefill-speculator graph only) — decide
  at M-mtp-4, mirror `PrefillSpeculatorCudaGraphManager`.
- GGUF checkpoints lack `mtp.*` → document as safetensors-only feature until
  we re-export GGUFs with the head.
- Acceptance-rate parity tolerance: define "within noise" from ≥3 oracle runs
  (benchmark-protocol.md reproduction rule).

## 9. SPEC-MTP → DONE (2026-07-26, `CLAIM-SPEC-MTP-DONE`, records-only)

MTP k=1 speculative decoding is **COMPLETE and gated**; `SPEC-MTP` moves from
`ACTIVE` to `DONE`. This is a records-only lifecycle transition (ZERO code) that
reconciles the row to the state landed by I5e/I6/I7; the closing commit is I7
`72f9fb1`.

**The I6-owed DONE items are both closed:**
- **(1) MIXED spec+non-spec `GdnBlockPaged` concurrency batch (I7).**
  `GdnBlockPagedMixedSpec` (`src/vllm/model_executor/models/qwen3_5.cpp:3301`)
  index_selects `mixed_qkv`/`a`/`b` by `spec_token_indx`/`non_spec_token_indx`
  into compact groups, runs the I5a spec conv+recurrence over the spec rows and
  the widened-cache-aware prefill conv+recurrence over the reclassified non-spec
  rows (upstream forces `num_decodes==0` when a spec row exists,
  `gdn_attn.py:243-251`), and index_copies the two cores back to their original
  positions — a 1:1 mirror of `qwen_gdn_linear_attn.py::_forward_core:1329-1576`.
  New row ops `vt::IndexSelect`/`vt::IndexCopy` (CUDA==CPU bit-exact, RED-first).
  Proven MODEL-INDEPENDENTLY bit-exact: `mixed == pure spec + pure prefill` on
  27B/35B dims (`tests/vllm/models/test_qwen3_5_gdn_spec_routing.cpp`), RED-first
  by a broken merge; compute-sanitizer 0 on the mixed step + op.
- **(2) Server-facing `--speculative-config` (I5d/I7).** Wired through the
  OpenAI server (`examples/server/main.cpp`), the example CLI
  (`examples/cli/main.cpp`) and the C ABI v6 (`src/capi/vllm_c.cpp` →
  `ParseSpeculativeConfigJson` in `vllm_engine_load`). Absent/NULL/empty ⇒
  byte-identical no-speculation engine; a malformed or unsupported document
  aborts load loudly (not silently ignored).

**The c>1 DONE criterion (user-ratified).** At concurrency > 1 the DONE bar is
the near-tie-distributional form (ours ∈ vLLM's batch-nondeterministic greedy
set) plus the SPEED delta — NOT strict token-exact. Strict c>1 token identity is
a proven bf16-batch-nondeterminism MODEL impossibility: the spec-OFF 27B greedy
itself differs at `max_num_seqs` 4-vs-1 on 2/3 short prompts with no spec
involved, and this affects vLLM identically. Correctness at c>1 therefore rests
on the model-independent bit-exact split/merge proof + acceptance parity, with
token-exact strict at c1.

**Gate evidence (I5e/I6/I7, on this exact code):** 27B three-way token-exact at
c1 (acceptance 16/16); c1 ours AT/ABOVE vLLM every axis (TPOT ~1.04x faster,
output tput +4.6%/+3.9%, acceptance 0.85/0.92 vs 0.838); c2-c8 ours
ON-PAR-OR-ABOVE vLLM spec-ON (output tput within ~±2%, ~1.5x spec speedup at
every concurrency, acceptance 0.84-0.92 vs 0.835); spec-OFF byte-identical SACRED
(27B 235/235, 35B 315/315, Coder 138/138); CUDA `-Werror` 0 warnings. Byte
identity of the DONE transition holds by construction (`git diff --stat` =
records only). Ledger: `.agents/parity-ledger.md` (2026-07-26 SPEC-MTP DONE row).

**Tracked follow-ons (own live rows, NOT DONE blockers):** the 35B
`Qwen3_5MoeMTP` full e2e three-way token gate (M-mtp-2,
`MODEL-SPEC-qwen3-5-mtp-qwen3-5-moe-mtp`) is now **CLOSED — `DONE` 2026-07-26**
(three-way token-exact 16/16 vs the live vLLM 0.25.0 oracle, acceptance 16/16
both sides, c1 spec-ON 1.19x/+16.3% vs spec-OFF; see §5 M-mtp-2 RESULT). MTP
spec-decode is therefore `DONE` on BOTH gate models (27B `Qwen3_5MTP` + 35B
`Qwen3_5MoeMTP`). Remaining spec-decode follow-on: `SPEC-DFLASH` (oracle-BLOCKED
on vLLM 0.25.0, vllm#40898).
