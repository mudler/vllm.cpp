# Spec: DFlash speculative decoding (task #51, from B5 scoping)

Block-diffusion drafting — the headline speculative method after MTP lands.
Derived from
[.agents/specs/spec-decode-scoping-2026-07-10.md](spec-decode-scoping-2026-07-10.md);
re-grounded against the pin (`/home/mudler/_git/vllm` @ `e24d1b24`) and the
live HF draft checkpoints (re-verified 2026-07-10). **Prerequisite:
[mtp-spec-decode.md](mtp-spec-decode.md) fully landed** — DFlash reuses its
scheduler plumbing, rejection sampler, draft-KV machinery, and (critically)
the GDN spec path, since BOTH gate targets are GDN hybrids (MTP spec §0).

## 1. What DFlash is (at our pin — it is fully in-tree)

arXiv 2602.06036 (ICML 2026, Z-Lab): a small (~0.3–0.5B) **block-diffusion
drafter** generates an entire 8–16-token block in ONE non-autoregressive
forward, conditioned on multi-layer hidden features of the target; standard
lossless rejection verify. Claims: >6× lossless on content-favorable
workloads, ~2.5× over EAGLE-3; measured WITH concurrency (B5): 35B-class on
B200 2.23–3.61× c1 → 2.03–2.89× c32. Community DGX-Spark container
(`github.com/AEON-7/vllm-dflash`) runs DFlash + NVFP4 targets on our exact
hardware (2–5× single-stream decode, content-dependent).

In-pin implementation (all mirrored 1:1):
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` (631 lines),
  `dflash/utils.py` (`load_dflash_model`, `get_dflash_causal`),
  `dflash/cudagraph.py`.
- Draft model: `vllm/model_executor/models/qwen3_dflash.py`
  (`DFlashQwen3ForCausalLM` :639, `DFlashQwen3Model` :323).
- Config: method `"dflash"` (`vllm/config/speculative.py:56,301`);
  auto-detected from a draft path containing "dflash" (`:789-790`);
  `parallel_drafting = True` (`:853-854`); draft hf_config wrapped as
  EAGLEConfig (`:822-840`).
- Scheduler: `num_lookahead_tokens = num_spec_tokens + 1`
  (`vllm/v1/core/sched/scheduler.py:247-252` — one extra slot because the
  anchor/bonus query occupies a position too).

## 2. Draft checkpoints for OUR exact gate models (verified live on HF)

| | `z-lab/Qwen3.6-35B-A3B-DFlash` | `z-lab/Qwen3.6-27B-DFlash` |
|---|---|---|
| arch / model_type | `DFlashDraftModel` / qwen3 | same |
| layers | 6 (5× SWA-4096 + 1 full) | 5 (4× SWA-2048 + 1 full) |
| hidden / heads / kv / head_dim | 2048 / 32 / 8 / 128 | 5120 / 32 / 8 / 128 |
| block_size (drafted block) | 16 (`dflash_config`) | 16 (top-level key) |
| mask_token_id | 248077 | 248070 |
| target_layer_ids (aux taps) | [1, 6, 11, 16, 22, 27, 32, 37] (8 taps, num_target_layers 40) | [1, 16, 31, 46, 61] (5 taps, num_target_layers 64) |
| dtype / vocab | bf16 / 248320 | bf16 / 248320 |
| rope_theta | 1e7 (full rotary, head_dim 128) | 1e7 |

Both are plain Qwen3-dense-style decoders (gate/up/down MLP, q/k/v/o + q/k
norm) — NO GDN, NO MoE in the draft. Downloads (188k / 46k) and the arXiv tag
confirm these are the official Z-Lab drafts for our checkpoints. The draft is
bf16 while targets are NVFP4 — exactly the combination the DGX-Spark community
container runs.

## 3. Anatomy at pin (what a port implements)

### 3.1 Per-step flow (`dflash/speculator.py:246-413`)
1. Target step finishes; propose() receives the target's per-token hidden
   states (aux-combined, §3.3) + `num_sampled/num_rejected/last_sampled`.
2. `prepare_dflash_inputs` (one Triton kernel, `:416-563`): for each request
   build a query block of `num_query_per_req = 1 + k` tokens (`:45`) — the
   **anchor** = bonus token (last sampled, or next prefill token for chunked
   prefill `:464-469`), then **k mask tokens** (`mask_token_id` embedding,
   `:497`); positions `last_valid_pos+1 …`; also computes context
   positions/slots from the target block table and per-mask-token sample
   indices. Rejected positions are excluded via `valid_ctx_end = ctx_end −
   num_rejected` (`:461-471`).
3. **Context-KV pre-insert** (`qwen3_dflash.py:471-568`
   `precompute_and_store_context_kv`): the target's hidden states for THIS
   step's tokens are projected to K/V for ALL draft layers in one fused GEMM
   (`_fused_kv_weight`), k-normed + RoPE'd in bulk, and written directly into
   the draft's KV caches at the context slots. So the draft never re-runs
   context tokens — its KV cache is filled from target features as the target
   computes them, step by step. Runs eagerly (context length varies), outside
   CUDA graphs (`:350-353` comment).
4. One draft forward over the `num_reqs × (1+k)` query tokens (uniform → FULL
   CUDA graph, `:369-378`); **non-causal in-block attention** for
   full-attention layers, causal for SWA layers
   (`qwen3_dflash.py:55-122` `_resolve_layer_attention`: full defaults
   non-causal, SWA defaults causal; per-layer `causal` metadata).
5. Sampling: each mask position t yields draft token t via the shared
   `sample_draft` (`:209-224`; Gumbel positions offset −2 so verify noise
   matches, `:212-215`). Anchor is not sampled (that's `sample_from_anchor =
   False` `:53-56`; the True variant is DSpark — out of scope).
6. Verify next step: the target runs the k+1-token query and the **SAME
   `RejectionSampler`** as MTP (`model_runner.py:1065-1079` — no DFlash-
   specific verify path). Scheduler rollback identical.

### 3.2 Draft KV cache
The draft's 5–6 attention layers are registered as extra attention layers →
own KV-cache group(s) (`speculator.py:163-169` draft layer names;
`dflash/speculator.py:125-164` resolves `draft_kv_cache_group_ids`, per-group
context slot buffers, and (for mixed SWA/full drafts like ours) a per-layer
group index). Slot mappings for query tokens are written into the shared
`BlockTables.slot_mappings` so captured graphs replay correctly (`:321-348`).

### 3.3 Aux hidden capture (target-side change!)
The drafter conditions on MULTI-LAYER target features: `target_layer_ids`
+1-shifted through the eagle3 interface
(`eagle/eagle3_utils.py:36-55` `get_eagle3_aux_layers_from_config` reads
`dflash_config.target_layer_ids`, adds 1) →
`model.set_aux_hidden_state_layers(...)`
(`qwen3_5.py:332-333` / `qwen3.py` SupportsEagle3). The target forward then
returns `(hidden, aux_hidden_states)` (`model_runner.py:1324-1332`), and the
drafter combines: `hidden = model.combine_hidden_states(cat(aux, dim=-1))`
(`dflash/speculator.py:285-291`; the fc lives in the draft:
`qwen3_dflash.py:377-395`, input = hidden_size × num_taps → hidden_size).

**For us**: our monolithic `qwen3_5.cpp` forward must tap the residual stream
at 8 (35B) / 5 (27B) layer boundaries into a `[T, H × taps]` buffer. Cheap
(copies only), but it touches the fused forward — keep the taps optional
(config-gated) so the non-spec hot path is unchanged.

### 3.4 What DFlash does NOT need
- No multi-step draft loop (one forward per block).
- No draft-side GDN (drafts are plain attention).
- No new rejection sampler, scheduler mechanics, or GDN target machinery
  beyond what MTP lands — but note the GDN state slots scale as k+1 per
  request (§5).

## 4. Shared-vs-new matrix (relative to the MTP spec)

| piece | status after MTP lands |
|---|---|
| scheduler: spec_token_ids, rollback, update_draft_token_ids | SHARED (only `num_lookahead_tokens = k+1`, `scheduler.py:247-252`) |
| rejection sampler (greedy + stochastic) | SHARED, unchanged |
| input-batch expansion, logits indices | SHARED |
| GDN spec path (slots, kernel, metadata split) | SHARED mechanism; k jumps 1 → 8/16 (§5 memory) |
| verify attention (varlen k+1 queries) | SHARED |
| draft KV group + block-table plumbing | SHARED pattern (MTP has 1 draft KV layer; DFlash has 5–6 + SWA groups) |
| aux hidden taps in target forward | NEW |
| DFlash draft model (Qwen3-dense w/ SWA + non-causal in-block attn) | NEW (small; our first non-causal attention path) |
| context-KV precompute (fused multi-layer KV proj + bulk RoPE + cache insert) | NEW |
| prepare_dflash_inputs kernel (mask-token blocks, context slots, sample maps) | NEW (port of `:416-563`) |
| separate mask embedding fallback | NEW-trivial (`qwen3_dflash.py:352-361`; our drafts use in-vocab mask tokens) |
| FULL CUDA graph for uniform 1+k query | NEW-ish (mirror `dflash/cudagraph.py`; uniform shape makes it the easy case) |

## 5. Sizing flags (must-measure before committing to defaults)

- **GDN state slots**: k+1 slots/request on the TARGET's GDN layers (MTP spec
  §3). At block 16: 27B ≈ 17 × 144 MiB ≈ **2.4 GiB per concurrent request**;
  35B ≈ 17 × 60 MiB ≈ 1.0 GiB. On GB10 (128 GB unified) this caps effective
  concurrency; vLLM pays the same cost — mirror first, then consider
  `block_size: 8` draft variants (the 35B card ships block 8/16 per B5) and
  measure the acceptance-vs-memory trade. This is the single biggest
  DFlash-on-hybrid risk and it is NOT visible in the B200 dense-model numbers.
- **Verify width**: 17-token queries per request per step through a 256-expert
  MoE (35B) unions many experts; GB10's compute:bandwidth ratio thins the B200
  margins (B5 caveat). Gate at both operating points as in the MTP spec §5.
- **Acceptance content-dependence**: same protocol as MTP (natural-language +
  code workloads; never synthetic).

## 6. Milestones

- **M-df-0 — draft model runs standalone**: load `z-lab/Qwen3.6-27B-DFlash`,
  implement the draft forward (SWA + non-causal full layers), context-KV
  precompute; parity vs the HF/vLLM draft on captured target features
  (token-exact block for greedy).
- **M-df-1 — 27B e2e greedy**: aux taps in target, prepare-inputs kernel,
  scheduler k+1 lookahead; token-exact 16/16 vs vLLM
  `--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash",
  "num_speculative_tokens":k}'` (mirror vLLM's default k for dflash = draft
  `block_size − 1` fallback chain, `speculative.py:865-880`), then A/B vs
  vLLM-with-DFlash on all axes.
- **M-df-2 — 35B e2e**: 8 aux taps + GDN slot memory measurement (§5) +
  concurrency sweep c1→c32.
- **M-df-3 — sampling + graphs**: stochastic verify (shared sampler), FULL CG
  for the uniform draft step.

## 7. Tests to port (protocol: .agents/test-porting.md)

| upstream | what it asserts | tier → ours |
|---|---|---|
| `tests/v1/spec_decode/test_dflash_lookahead.py` (`test_dflash_prefill_reserves_lookahead_blocks` :98, `test_dflash_first_prefill_query_window_fits_allocated_blocks` :117, `test_dflash_drafter_window_reserves_bonus_token` :134) | k+1 lookahead slot accounting: prefill reserves blocks for the query window incl. bonus | T-unit → `tests/vllm/v1/spec_decode/test_dflash_lookahead.cpp` |
| `tests/v1/e2e/spec_decode/test_spec_decode.py::test_dflash_acceptance_rates` (:1323-1345, fixture `dflash_config` :1323) | e2e acceptance-rate floor with a real DFlash draft | T-e2e (nightly dgx) → paged-engine spec config with the z-lab drafts; we additionally gate 16/16 greedy token-exactness vs the oracle (stricter, per gates.md) |
| shared rejection-sampler suite (see MTP spec §6: `tests/v1/spec_decode/test_rejection_sampler_utils.py`, `tests/v1/sample/test_rejection_sampler.py`) | verify correctness independent of drafter | already ported under the MTP spec; DFlash adds parametrizations with k=block−1 |
| `tests/v1/attention/test_gdn_metadata_builder.py` spec cases | GDN split at k≫1 | already ported under the MTP spec; extend params to k=15 |
| draft-model unit coverage (upstream has no dedicated `qwen3_dflash` unit test at pin — model correctness is covered via the e2e acceptance test) | non-causal in-block attention, SWA layers, context-KV precompute numerics | T-parity → golden-dump tests vs the HF draft implementation (`tools/parity/` — same discipline as our kernel goldens); cite `qwen3_dflash.py` in the test header |
| `tests/v1/spec_decode/test_max_len.py`, `test_dynamic_sd.py` (generic spec guards) | max-model-len clamping with lookahead; dynamic spec on/off | T-unit, port the cases that apply to draft-model methods; SKIP dynamic-SD until we mirror that feature |

## 8. Open questions

- vLLM's default `num_speculative_tokens` for these drafts (block 16 → k=15?
  or the block-8 variant): resolve by running the oracle with the z-lab cards'
  recommended config and mirroring what `SpeculativeConfig` resolves
  (`speculative.py:865-880` n_predict/num_lookahead chain) — never hand-pick.
- Whether the 27B draft's top-level `block_size: 16` (vs in-`dflash_config`
  for the 35B) resolves identically through `EAGLEConfig` wrapping — check at
  M-df-0 with the oracle.
- trust_remote_code: the drafts ship `custom_code` (`dflash.DFlashDraftModel`)
  — vLLM uses its in-tree `qwen3_dflash.py` instead; we mirror the in-tree
  model, so no remote code in our engine.
- DSpark (`dspark/`, deepseek drafts) shares this scaffolding — explicitly out
  of scope; revisit only if a Qwen3.6 dspark draft appears.

Sources: pin files cited inline · arxiv.org/abs/2602.06036 ·
z-lab.ai/projects/dflash · HF z-lab/Qwen3.6-{35B-A3B,27B}-DFlash (configs
fetched 2026-07-10) · github.com/AEON-7/vllm-dflash · B5 scoping report.
