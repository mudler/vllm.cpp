# Speculative decoding scoping (2026-07-10) — vLLM V1 map, MTP, DFlash, ranked route

Pin `/home/mudler/_git/vllm` @ e24d1b24. Full agent report distilled; file:line cites verified.

## 1. The 1:1 mirror target

Our C++ runner mirrors MRV2 → the spec stack to port is **`vllm/v1/worker/gpu/spec_decode/`**
(NOT the legacy `vllm/v1/spec_decode/`): `speculator.py:31,74` (BaseSpeculator/DraftModel),
`autoregressive/speculator.py:30` (propose(): draft prefill REUSES the target's attn metadata
+ slot mappings `:222-234`; k−1 one-token decode steps `_multi_step_decode:371`),
`mtp/speculator.py:12` (**18-line subclass**), `rejection_sampler.py:43,101` (greedy path =
plain argmax-equality), `dflash/speculator.py:31`. Scheduler hooks = exactly our stubs:
`num_lookahead_tokens` (`scheduler.py:245-252`; dflash = k+1), `spec_token_ids` (`:593-609`),
rejection rollback `num_computed_tokens -= num_rejected` (`:1585-1612`). Drafter owns its own
KV layer(s); Mamba/hybrid targets get `MambaSpec.num_speculative_blocks`.

## 2. MTP: both gate checkpoints SHIP heads (verified against HF safetensors)

- **35B** (`Qwen3_5MoeForConditionalGeneration`): `mtp_num_hidden_layers: 1`, 19 `mtp.*`
  tensors (fc [2048,4096], pre_fc_norms, one full-attn + full-MoE decoder layer), BF16 ~1.69GB.
- **27B**: 15 `mtp.*` tensors (fc [5120,10240], one dense full-attn layer), BF16 ~0.8GB.
- vLLM auto-resolves `qwen3_5(_moe)` → `qwen3_5_mtp` (`config/speculative.py:480-489`);
  model `qwen3_5_mtp.py:63,129-165` (fc(cat[norm(embed),norm(hidden)]) → one decoder layer →
  norm → SHARED lm_head). Gotcha: `mtp.fc` is bf16-unquantized in NVFP4 checkpoints (:86-94).

**C++ port needs:** stop skipping `mtp.*` (`qwen3_5_dense_weights.cpp:209`,
`qwen3_5_weights.cpp:328`); the hidden-state tap already exists (logits-gather rows,
`qwen3_5.cpp:3074-3094`); MTP forward reuses our decoder-layer blocks; flip the scheduler
stubs (`scheduler.h:217`, `output.h:148`, rollback TODO `scheduler.cpp:406`); greedy verify =
small argmax-equality at `runner.cpp:704`; verify-attention is free (varlen prefill kernels
route `num_tokens>num_reqs`, `cuda_paged_attn.cu:2317`); `allocate_slots` already honors
lookahead. **The one hard kernel gap (35B only): GDN multi-token decode** — our backend
hardwires `num_spec_decodes=0` (`gdn_attn.h:74-75`); upstream snapshots per-draft-position
recurrent states and selects by acceptance (`gdn_attn.py:47-66`, fla `IS_SPEC_DECODING`,
`fused_sigmoid_gating.py:20,106`). The 27B (pure attention) has no such problem → port there first.

## 3. DFlash = Block Diffusion for Flash Speculative Decoding

arXiv 2602.06036 (ICML 2026, Z-Lab): a ~0.4B block-diffusion drafter generates a whole 8-16
token block in ONE forward, conditioned on multi-layer target features; lossless rejection
verify. >6× lossless claims; 2.5× over EAGLE-3. **Already in our pin** (`dflash/speculator.py`,
`qwen3_dflash.py`, documented method). **Trained drafts exist for OUR gate models**:
z-lab/Qwen3.6-35B-A3B-DFlash (0.4B, block 8/16, target_layer_ids [1,6,11,16,22,27,32,37]) and
z-lab/Qwen3.6-27B-DFlash. Measured WITH concurrency: 35B on B200 2.23-3.61× c1 → 2.03-2.89×
c32; community DGX-Spark container (AEON-7/vllm-dflash) runs DFlash+NVFP4 on our hardware
(2-5× decode single-stream, content-dependent).

## 4. Ranked route

1. **MTP first** (k=1, greedy, 27B → 35B): heads are FREE in-checkpoint, exercises 100% of the
   shared plumbing every later method reuses. ~1.15-1.35× decode at conc16/32 (community: +27.5%
   k=1 on the 35B).
2. **DFlash second (the headline)**: published drafts, in-pin implementation, best
   high-concurrency retention. Adds: aux hidden capture at 5-8 layers, the draft model class,
   context-KV pre-insert, non-causal in-block attention, lookahead k+1.
3. ngram: plumbing smoke test only. 4. EAGLE: dominated (no public Qwen3.6 heads) — skip.

**Milestones:** M-spec-1 = MTP k=1 greedy 27B on GB10, token-exact vs vLLM
`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`. M-spec-2 = the GDN spec
path (hardest single piece; prerequisite for 35B + DFlash-35B). M-spec-3 = k>1 + Gumbel
rejection. M-spec-4 = DFlash.

**Honest caveats:** batch decay (4.0×→1.9× c1→c32 on a 9B); 35B MoE verify unions more
experts/token; GB10 compute:bandwidth thins margins vs B200; acceptance is content-dependent
(prose ~2.0/15 vs code ~5.5/15 — synthetic gate prompts will show ~nothing; A/B must use
natural-language workloads); once enabled, the honest denominator = vLLM WITH the same spec
config; at in1024/out128, 2× decode ≈ 1.3× e2e.

Sources: arxiv.org/abs/2602.06036 · z-lab.ai/projects/dflash · github.com/z-lab/dflash ·
HF z-lab/Qwen3.6-{35B-A3B,27B}-DFlash · github.com/AEON-7/vllm-dflash · llama.cpp PR #22105.
