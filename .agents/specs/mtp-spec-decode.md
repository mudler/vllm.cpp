# Spec: MTP speculative decoding (task #51, from B5 scoping)

Implementation-ready spec for Multi-Token-Prediction speculative decoding on
both gate checkpoints. Derived from
[.agents/spec-decode-scoping-2026-07-10.md](../spec-decode-scoping-2026-07-10.md);
every claim re-grounded against the pin (`/home/mudler/_git/vllm` @ `e24d1b24`)
and the actual checkpoints on `dgx.casa`. Companion spec:
[dflash-spec-decode.md](dflash-spec-decode.md) (depends on this one).

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
  27B loader already declares this: `qwen3_5_dense_weights.cpp:209`).
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

- **M-mtp-0 — weights + head parity (no engine changes).** Load `mtp.*` on
  both models; run the MTP head standalone on captured target hidden states;
  token-exact vs a vLLM-oracle script driving `Qwen3_5MTP` on the same inputs.
  Gate: argmax match on ≥16/16 positions across a natural-language prompt set,
  both checkpoints.
- **M-mtp-1 — 27B k=1 greedy e2e (includes the GDN spec path — see §0).**
  Scheduler plumbing + prepare-prefill kernel + greedy rejection + GDN slot
  mechanism (k+1=2 slots). Gate: token-for-token 16/16 vs
  `vllm --speculative-config '{"method":"mtp","num_speculative_tokens":1}'`
  greedy, AND acceptance-rate telemetry within noise of vLLM's on the same
  prompts. Then A/B decode throughput + TPOT (expected from B5/community:
  ~1.15–1.35× decode at conc16/32; +27.5% k=1 reported on the 35B-class).
- **M-mtp-2 — 35B k=1 greedy.** Adds only the MoE MTP layer (reuses our MoE
  blocks) — GDN path identical. Same gates. Caveat: verify unions experts
  across k+1 tokens/request (more experts touched per step — measure, don't
  assume the 27B speedup transfers).
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
