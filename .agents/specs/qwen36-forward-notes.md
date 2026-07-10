# Qwen3.6 forward assembly + oracle recipe — M0.9 Task 1 notes

The M0.9 milestone assembles the Qwen3.6 forward (embed -> N decoder layers ->
norm -> lm_head -> logits) from the vt ops and proves logits + greedy-decode
parity vs a pinned oracle. Task 1 (this doc) DE-RISKS the oracle: it establishes
exactly how to obtain reference logits + per-layer hidden states from the REAL
model on dgx BEFORE any forward code is written. Everything here was determined
empirically on dgx (GB10, 119 GiB unified) 2026-07-03; every config value was
read from the real `config.json` / checkpoint, not memory.

Dump script: `tools/parity/dump_qwen36.py`. Goldens:
`tests/parity/goldens/qwen36_*_{27b,35b}`.

---

## 0. HEADLINE FINDINGS (read these first)

1. **The two gate checkpoints are DIFFERENT architectures.** The plan/roadmap
   assumed "27B and 35B are both MoE gates". They are not:
   - `unsloth/Qwen3.6-27B-NVFP4` = **`Qwen3_5ForConditionalGeneration`**,
     `model_type qwen3_5` / text `qwen3_5_text` — a **DENSE** multimodal model
     (dense SwiGLU MLP, `intermediate_size 17408`, NO experts), 64 layers,
     hidden 5120. Quantized **compressed-tensors NVFP4 W4A4** (weights AND
     activations 4-bit).
   - `nvidia/Qwen3.6-35B-A3B-NVFP4` = **`Qwen3_5MoeForConditionalGeneration`**,
     `model_type qwen3_5_moe` / text `qwen3_5_moe_text` — the **MoE** model
     (256 experts, top-8, `moe_intermediate_size 512`, shared expert 512),
     40 layers, hidden 2048. Quantized **modelopt NVFP4 W4A16** (weight-only,
     activations bf16) per `.agents/specs/moe-semantics.md` §8.
   - **Implication for the plan:** Task 4's forward dispatches on `model_type`
     and builds the MoE block only for `qwen3_5_moe`. So a full-model parity run
     **on the 27B does NOT exercise the MoE path at all** — it validates GDN +
     dense-attn + dense MLP + embed/norm/lm_head. To gate the MoE forward
     end-to-end you MUST run the **35B**. Both are dumped here
     (`_27b` = dense, `_35b` = MoE) so Task 5 can gate the right paths.

2. **Both models are multimodal** (`*ForConditionalGeneration`, with a
   `vision_config` + `model.visual.*` tower). The language stack lives under
   `model.language_model.*`. A text-only prompt runs the language model; vLLM
   still loads the vision tower (and briefly runs it during the dummy/profile
   pass), so it must be present but is irrelevant to our text forward.

3. **Quant path is the real NVFP4 GEMM, not bf16-dequant.** The 27B runs
   **cutlass FP4 W4A4** (FlashInfer `mm_fp4`, activations dynamically quantized
   to fp4). The plan's C++ approach (DequantNvfp4ToBf16 + bf16 Matmul) is a
   HIGHER-precision path than the 27B oracle -> per-layer/full-model logits will
   differ by MORE than bf16 rounding on the fp4 layers (attn qkv/o, MLP). The
   **35B is modelopt W4A16** (weight-only 4-bit, bf16 activations) which matches
   the dequant-to-bf16 approach far more closely -> the 35B is the better parity
   target for our planned kernels. Recorded as a Task 5 tolerance concern.

4. **Oracle engine = pip vLLM 0.24.0, NOT the pinned checkout** (documented
   deviation — see §3 for why the pinned overlay is blocked). pip 0.24.0 is the
   same release family as pin `e24d1b24`; it loads these exact checkpoints and
   greedy-decodes coherently. Manifests record `oracle.source = "pip-vllm:0.24.0"`
   + `pin_reference = "e24d1b24"`.

---

## 1. Config values (read from the REAL config.json on dgx, 2026-07-03)

### 27B — `unsloth/Qwen3.6-27B-NVFP4` (snapshot 890bdef7), DENSE, `qwen3_5_text`

| field | value |
|---|---|
| architectures | `Qwen3_5ForConditionalGeneration` |
| hidden_size | 5120 |
| num_hidden_layers | 64 |
| intermediate_size (dense MLP) | 17408 |
| vocab_size | 248320 |
| tie_word_embeddings | **false** (separate `lm_head.weight`) |
| num_attention_heads | 24 |
| num_key_value_heads | 4 (GQA ratio 6) |
| head_dim | 256 |
| partial_rotary_factor | 0.25 -> rotary dim = 64 |
| attn_output_gate | **true** (q_proj emits 2x = q + gate) |
| rope: theta / mrope | 1e7, `mrope_interleaved: true`, `mrope_section [11,11,10]` |
| rms_norm_eps | 1e-6 |
| full_attention_interval | 4 |
| layer_types | 64 entries: `[LA,LA,LA,FA] x 16` (LA=linear_attention, FA=full_attention) |
| GDN: Hk / Hv / Dk / Dv / conv | 16 / **48** (GQA ratio 3) / 128 / 128 / 4 |
| GDN output_gate_type | `swish` -> silu gating |
| mamba_ssm_dtype | float32 |

### 35B — `nvidia/Qwen3.6-35B-A3B-NVFP4` (snapshot 491c2f1e), MoE, `qwen3_5_moe_text`

| field | value |
|---|---|
| architectures | `Qwen3_5MoeForConditionalGeneration` |
| hidden_size | 2048 |
| num_hidden_layers | 40 |
| num_experts / top_k | 256 / 8 |
| moe_intermediate_size / shared_expert_intermediate_size | 512 / 512 |
| vocab_size | 248320 |
| tie_word_embeddings | false |
| num_attention_heads / num_key_value_heads / head_dim | 16 / 2 / 256 |
| partial_rotary_factor | 0.25 (rotary dim 64) |
| attn_output_gate | true |
| GDN: Hk / Hv / Dk / Dv / conv | 16 / 32 / 128 / 128 / 4 |
| GDN output_gate_type | absent -> silu; rms_norm_eps 1e-6 |
| layer_types | 40 entries: `[LA,LA,LA,FA] x 10` |

Both: `layer_types.index("linear_attention") == 0`,
`layer_types.index("full_attention") == 3` (the layer indices dumped).

## 1b. Checkpoint tensor naming + layout (27B, read from model.safetensors)

Single-shard `model.safetensors` (24.57 GiB), 2111 tensors, `format: pt`.
Names are under `model.language_model.*`; `lm_head.weight` at top level.

- **bf16, load direct** (in the quant `ignore` list):
  `model.language_model.embed_tokens.weight [248320,5120]`,
  `model.language_model.norm.weight [5120]`, `lm_head.weight [248320,5120]`,
  and per-layer GDN in-projections + conv + scalars/norms:
  `linear_attn.in_proj_qkv [10240,5120]` (= 2*Hk*Dk + Hv*Dv = 4096+6144),
  `in_proj_z [6144,5120]` (Hv*Dv gate), `in_proj_b [48,5120]`,
  `in_proj_a [48,5120]`, `conv1d.weight [10240,1,4]`, `A_log [48]`,
  `dt_bias [48]`, `linear_attn.norm.weight [128]` (RMSNormGated over Dv),
  and every `input_layernorm.weight` / `post_attention_layernorm.weight [5120]`,
  and full-attn `self_attn.q_norm.weight [256]` / `k_norm.weight [256]`.
- **compressed-tensors NVFP4 W4A4** (4 tensors per projection — DIFFERENT from
  the 35B's modelopt scheme in moe-semantics §8): `*.weight_packed U8 [out,in/2]`,
  `*.weight_scale F8_E4M3 [out,in/16]`, `*.weight_global_scale F32 [1]`,
  `*.input_scale`... actually `*.input_global_scale F32 [1]` (present because
  activations are dynamically fp4-quantized — W4A4). Quantized projections:
  GDN `linear_attn.out_proj` ([5120,3072] packed), full-attn
  `self_attn.{q,k,v,o}_proj` (q_proj [12288,2560] = 24*256*2 for the output
  gate; k/v_proj [1024,2560] = 4*256; o_proj [5120,3072]), and dense MLP
  `mlp.{gate,up,down}_proj` (gate/up [17408,2560], down [5120,8704]).
- **compressed-tensors dequant** (Task 3, DISTINCT from modelopt): the naming is
  `weight_packed`/`weight_scale`/`weight_global_scale`, and activations are also
  fp4 (input_global_scale present). This is a SEPARATE dequant contract from the
  35B modelopt path — record precisely in Task 3 before implementing.

## 1c. GDN in_proj fusion (both models)

Checkpoints ship the GDN input projections SEPARATELY (`in_proj_qkv`,
`in_proj_z`, `in_proj_b`, `in_proj_a`); the pinned `Qwen3_5Model` fuses them
into qwen3-next's `in_proj_qkvz` / `in_proj_ba` params via
`hf_to_vllm_mapper` (qwen3_5.py:201-211):
`.in_proj_qkv -> .in_proj_qkvz shards (0,1,2)`, `.in_proj_z -> .in_proj_qkvz 3`,
`.in_proj_b -> .in_proj_ba 0`, `.in_proj_a -> .in_proj_ba 1`. Task 3's loader
must replicate this fusion (or keep them separate and adapt the GDN op call).

---

## 2. THE WORKING RECIPE (what to run)

**Oracle = the pip vLLM 0.24.0 v1 offline engine, driven in-process, with
forward hooks.** This is the pragmatic answer to the plan's Task 1 question
"construct the model standalone vs use the offline LLM API": constructing
`Qwen3_5*ForCausalLM` standalone needs a full VllmConfig + per-layer attention
metadata + KV caches + mamba state for BOTH layer types — large and fragile.
The offline `LLM()` engine builds all of that automatically. We get per-layer
tensors by registering `torch` forward hooks on the in-process model and read
logits from the model's own bf16 lm_head.

Key knobs / gotchas (all learned the hard way):

- **`VLLM_ENABLE_V1_MULTIPROCESSING=0`** — forces the engine core in the SAME
  process so hooks (and the module-global `CAPT` dict they fill) are reachable.
  Without it the model lives in a spawned EngineCore subprocess.
- Script must be under `if __name__ == "__main__":` (v1 spawns helpers; a
  bare module-level `LLM()` triggers a freeze_support crash even with MP off).
- **`ninja` must be on `PATH`** (`PATH=~/venvs/vllm-oracle/bin:$PATH`). The
  first quantized forward JIT-builds + autotunes the FlashInfer cutlass FP4
  GEMM (`mm_fp4`); it shells out to `ninja` and dies with `FileNotFoundError:
  ninja` if the venv bin isn't on PATH (ninja is pip-installed but not linked
  to a PATH dir). First build + autotune ~a few minutes, then cached.
- **`gpu_memory_utilization`** is measured against TOTAL (119.63 GiB), and the
  check requires `free >= util*total`. With ~48 GiB already used the default
  0.9 fails; 0.5 works. Also: kill straggler python procs from timed-out runs
  (they hold GPU memory) before relaunching.
- `LLM(enforce_eager=True, max_model_len=256, max_num_seqs=1, dtype="bfloat16")`.
  Load: 27B ~2m23s weights + FP4 autotune (~5 min cold, ~1 min warm-cached RAM).
- Reach the in-process model with **`llm.apply_model(func)`** (runs `func(model)`
  on the worker, same process). Submodule paths:
  `model.language_model.model.embed_tokens` / `.layers[i]` / `.norm`, and
  `model.language_model.lm_head`.

### Per-layer capture (forward hooks)

Layer forward is `layer(positions=, hidden_states=, residual=) -> (h, r)` with
the fused add-RMSNorm residual thread (qwen3_next.py:474). `layer_scale` is
FALSE for these configs (skip those branches). The **residual stream** is what a
standalone C++ layer needs:

- stream INTO layer i = `hidden_states` (i==0, residual None) or
  `hidden_states + residual` (i>0). Captured in a `forward_pre_hook`.
- stream OUT of layer i = `out_hidden + out_residual`. Captured in a
  `forward_hook`. This is exactly the value fed to layer i+1's input_layernorm,
  and to the final `norm(hidden, residual)` (qwen3_next.py:91).

Sums done in f32 (the fused kernel accumulates residual in f32 before rounding;
f32 keeps the pre-round value). A C++ layer test is: `stream_in -> layer(loaded
weights, positions) -> stream_out`.

- **Embed** golden: hook `embed_tokens` output `[T, H]`.
- **Final norm** golden: pre-hook input stream `[T,H]` -> post-hook normed
  `[T,H]` (norm is called positionally `norm(hidden, residual)`).
- **Logits**: NOT captured during generate (vLLM only computes last-token logits
  when sampling). Instead compute `F.linear(norm_out_bf16, lm_head.weight)`
  using the model's own **unquantized bf16** `lm_head.weight` `[vocab,H]` — this
  reproduces `compute_logits` -> `LogitsProcessor` exactly (scale 1.0, no cap).

### Prefill-only guard

`generate(temperature=0, max_tokens=16)` runs one prefill (token dim == prompt
len) then decode steps (token dim == 1). Hooks store **only** when the tensor's
first dim == `PROMPT_LEN` and the key is unset -> captures the prefill pass and
ignores decode steps. (Prompt len must be > 1; ours is 9.)

### Fixed prompt

`"The capital of France is Paris, and the"` -> ids
`[760, 6511, 314, 9338, 369, 11751, 11, 321, 279]` (T=9, `add_special_tokens=True`,
no chat template). Greedy continuation (27B, verified coherent):
`" capital of Germany is Berlin.\nThe capital of France is Paris, and the"`
(ids `[6511,314,9564,369,19241,13,198,760,6511,314,9338,369,11751,11,321,279]`).
Last-prefill-position argmax == first greedy token (asserted).

### mrope positions

`positions` is `[3, T]` i64 (mrope 3 sections). For a TEXT-only prompt the 3
rows are EQUAL (sequential 0..T-1). **This is the crux of M0.9's RoPE scope:**
because all three mRoPE position rows equal the token position, mRoPE collapses
EXACTLY to partial NeoX RoPE on `positions[0]` with rotary dim 64. The pinned
model itself gates its fused rotary path on `is_neox_style AND text_only` and
applies `pos = positions[0]` in that case — i.e. for text-only single-sequence
the section-split (`mrope_section [11,11,10]`) is never exercised; every section
uses the same position, so it is identical to feeding `positions[0]` into a
plain partial NeoX RoPE. **Therefore Task 2 REUSES `kRopeNeox` on
`positions[0]` (rotary_dim 64) — NO true section-split mRoPE.** The
`mrope_section [11,11,10]` / `mrope_interleaved: true` values are recorded here
as a note for completeness but are **NOT needed for M0.9**; true section-split
mRoPE is multimodal-only (distinct positions per section) and is DEFERRED to
when the vision/multimodal path lands. Positions are dumped in the layer
goldens.

---

## 3. WHAT WAS TRIED AND BLOCKED (pinned-oracle attempt)

The plan requires "pinned-checkout oracle only". The pinned tree is at
`~/work/vllm-pin` (`e24d1b24`, pure git source, no compiled artifacts). Attempts:

1. **Isolated pinned layer modules (dump_gdn/dump_moe stub pattern)** — NOT
   viable for whole decoder layers: they need VllmConfig + attention-metadata +
   KV/mamba caches for both layer types. That plumbing is precisely what the
   engine provides; re-stubbing it is a milestone-sized effort.
2. **Pinned Python overlaid on pip's compiled kernels** (copy pip's complete
   `vllm/` then overlay all pinned `*.py`, keeping pip's vendored build dirs).
   Import of `vllm` succeeds, but engine startup fails: pinned
   `v1/attention/backends/fa_utils.py` imports
   `compile_flash_attn_varlen_func_from_specs` from `vllm.vllm_flash_attn`,
   which pip 0.24.0's **compiled** `vllm_flash_attn` does not export. **The
   pinned Python (post-0.24.0-tag) expects a NEWER vendored/compiled API than
   the pip 0.24.0 wheel ships.** Running the pinned code faithfully needs a
   matching from-source build of vLLM at `e24d1b24` (compile `_C`/`_moe_C`/
   `vllm_flash_attn`) — large, and out of scope for de-risking.
   (Earlier cascade also revealed the pin lacks build-vendored Python:
   `third_party/flashmla/flash_mla_interface.py`, `vllm_flash_attn/layers/*`.)

**Resolution (documented deviation):** use **pip vLLM 0.24.0** as the
end-to-end + per-layer oracle. Justification: (a) same release family as the pin
(`e24d1b24` is post-`0.24.0`-tag dev); (b) it loads these exact NVFP4
checkpoints and **greedy-decodes coherently** (27B: "France is" -> " Paris");
(c) the pin's diffs vs pip in `qwen3_5.py`/`qwen3_next.py` are weight-loader
plumbing (the in_proj fusion mapper — which pip also handles, since output is
correct) + MoE FSE helpers (unused by the dense 27B). Manifests carry
`oracle.source="pip-vllm:0.24.0"`, `pin_reference="e24d1b24"`.

**Follow-up option (if a bit-exact pinned oracle is later required):** build
vLLM from `~/work/vllm-pin` into a dedicated venv (`pip install -e .` with the
CUDA toolchain) so the compiled kernels match the pinned Python, then re-run
`dump_qwen36.py` under it — the script auto-labels `source="pinned:..."` when
`vllm.__file__` contains `pinenv`. Not needed for M0.9 layer iteration.

---

## 4. GOLDENS PRODUCED (`tests/parity/goldens/qwen36_*`)

Per model tag (`27b` dense, `35b` MoE), 5 cases, ~1.3 MiB/model (budget 2 MiB):

| case | tensors | tol | purpose |
|---|---|---|---|
| `qwen36_embed_<tag>` | token_ids[T], embed[T,H] f32 | 1e-3 | embed lookup |
| `qwen36_gdn_layer_<tag>` | hidden_in[T,H], out[T,H] f32, positions[3,T] | 5e-2 | full linear_attention decoder layer (GDN + MLP/MoE), layer 0 |
| `qwen36_fullattn_layer_<tag>` | hidden_in, out, positions | 5e-2 | full full_attention decoder layer (qk-norm + partial mrope + GQA + output gate + MLP/MoE), layer 3 |
| `qwen36_norm_<tag>` | hidden_in[T,H], out[T,H] f32 | 1e-3 | final RMSNorm |
| `qwen36_logits_<tag>` | token_ids, argmax[T] i32, topk_values[T,1000] f32, topk_indices[T,1000] i32, greedy_ids[16] i32 | argmax exact; logits 5e-2 | full-model reference + greedy decode |

Tolerances: bf16 layers (embed/norm) tight (1e-3); layers containing fp4 GEMM
(gdn/full-attn/logits, 27B) loose (5e-2) because the oracle runs cutlass FP4
while our C++ plans bf16-dequant (§0.3). Layer indices come from
`layer_types.index(...)`. Logits are argmax + top-1000 (full [T,248320] f32 =
~9 MB would blow budget); greedy match is the real M0 gate.

Re-dump command (per model):

    ssh dgx.casa 'cd ~/work/vllm.cpp && VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      PATH=~/venvs/vllm-oracle/bin:$PATH ~/venvs/vllm-oracle/bin/python \
      tools/parity/dump_qwen36.py --model <snapshot_dir> --tag <27b|35b> \
      --out tests/parity/goldens'

---

## 5. Task 2 — dense causal attention op (EXACT formula, cites)

Source of truth: pinned `vllm/model_executor/models/qwen3_next.py::Qwen3NextAttention`
(the 35B `Qwen3_5MoeForConditionalGeneration` reuses this exact class for its
1-in-4 `full_attention` layers; forward math is byte-identical to the pin, §0.4).
Lines cited are in that file.

**The full-attn layer forward** (`Qwen3NextAttention.forward`, L381-392):
1. `qkv, _ = self.qkv_proj(hidden_states)`  — QKV+gate projection (Task 4/caller).
2. `q, k, v, gate = self._project_qkv_gate(qkv, positions)`  — the op's core.
3. `attn_output = self.attn(q, k, v)`  — causal scaled-dot-product attention.
4. `if gate is not None: attn_output = attn_output * torch.sigmoid(gate)`  — output gate.
5. `output[:], _ = self.o_proj(attn_output)`  — output projection (Task 4/caller).

**`_project_qkv_gate` eager path** (the reference; the fused Triton kernel L336-357
is a CUDA-only optimization that is numerically the same, gated on
`attn_output_gate AND is_neox_style AND is_cuda AND text_only`, L318-323):
- **Gate split** (L359-367): `attn_output_gate=True` (config, notes §1), so
  `q_gate, k, v = qkv.split([q_size*2, kv_size, kv_size], -1)`. Then
  `q_gate.view(T, num_heads, 2*head_dim)`, `torch.chunk(q_gate, 2, dim=-1)` →
  `q` = first `head_dim` per head, `gate` = second `head_dim` per head. So per
  q-head the projection lays out `[q(head_dim) | gate(head_dim)]` contiguously.
- **qk-norm** (L372-377): `q = self.q_norm(q.view(-1, num_heads, head_dim))`,
  `k = self.k_norm(k.view(-1, num_kv_heads, head_dim))`. `q_norm`/`k_norm` are
  `Qwen3NextRMSNorm = GemmaRMSNorm` (import alias L27) over `head_dim` (=256),
  `eps=rms_norm_eps` (1e-6). **GEMMA-style** → weight applied as `(1 + w)`, math
  in f32 (the fused path passes `q_norm.weight.float() + 1.0`, L347-348, confirming
  gemma). Per-(token,head) normalization over the head_dim axis.
- **RoPE** (L378): `q, k = self.rotary_emb(positions, q, k)`. `rotary_emb =
  get_rope(head_size=256, rope_parameters=..., ...)`. `partial_rotary_factor 0.25`
  → **rotary_dim = 64**, `is_neox_style=True`, partial NeoX (first 64 dims of each
  head rotate, dims 64..255 pass through). For text-only single-seq, `positions`
  is mRoPE `[3,T]` with all three rows equal, so this is EXACTLY partial NeoX RoPE
  on `positions[0]` (§2 proof). → **reuse `kRopeNeox` on `positions[0]`,
  rotary_dim 64.**
- **v** is NOT normed and NOT roped.

**Attention** (`self.attn`, L294-309, an `Attention(num_heads, head_dim, scaling,
num_kv_heads=..., attn_type=DECODER)`): causal scaled-dot-product.
- `scaling = head_dim ** -0.5` (L254) = `256 ** -0.5` ≈ 0.0625.
- GQA: `num_heads=16` q-heads, `num_kv_heads=2` kv-heads (35B, notes §1). q-head
  `h` reads kv-head `h // (num_heads // num_kv_heads)` = `h // 8`.
- `scores[i,j] = scaling * (q[i,h] · k[j,g])` for `j <= i` else `-inf`; row
  softmax (f32); `out[i,h] = Σ_j softmax_j · v[j,g]`.

**Output gate** (L390-391): `attn_output * sigmoid(gate)`, elementwise over
`[T, num_heads*head_dim]`. `gate` is the pre-sigmoid second half of the per-head
q-projection (above). Sigmoid is on the RAW gate (no norm/rope on gate).

### op decomposition (matches the M0.8 MoE precedent: thin new kernel + compose)

The only piece composition cannot express is the **causal GQA scaled-dot-product
attention** → new op **`vt::Attention` / `OpId::kAttention`** (CPU+CUDA, f32
softmax accumulation). qk-norm reuses `vt::RmsNorm` (gemma) over `head_dim`; RoPE
reuses `vt::RopeNeox` (rotary_dim 64); the gate-split (slicing the fused q/gate)
and the `sigmoid(gate)` multiply are elementwise and handled by the caller
(Task 4 layer / the parity runner), exactly like the MoE shared-expert sigmoid is
composed in `RunMoeBlock`. So Task 2 does NOT deliver the full LAYER (the
qkv/o projections are Task 4) → **`qwen36_fullattn_layer` STAYS in
PendingRunnerOps** (Task 4 removes it). Task 2's own op golden is
`dense_attention_*` (dedicated, NOT in the allowlist → its runner runs in CI).

**`kAttention` contract:** `query [T,Hq,D]`, `key [T,Hk,D]`, `value [T,Hk,D]`,
`out [T,Hq,D]`; `AttentionArgs{scale, causal}`; `Hq % Hk == 0`; f32 or bf16 in,
f32/bf16 out, f32 math. q/k arrive POST-norm+POST-rope; v raw.

### `dense_attention_*` golden (op-level, `tools/parity/dump_ops.py::dump_attention`)

Oracle: vLLM `GemmaRMSNorm.forward_native` (qk-norm) + `get_rope(...).forward_native`
(partial NeoX, same API as `dump_rope`) + pure-torch causal GQA sdpa (f32) +
`torch.sigmoid` gate — every piece pinned-equivalent. Two f32 cases: `_small`
(T=6,Hq=4,Hk=2,D=8,rotary 4) and `_realdims` (T=9,Hq=4,Hk=2,D=256,rotary 64,
theta 1e7). Tensors: `q,gate,k,v` (pre-norm inputs), `positions[T]`,
`q_norm_weight,k_norm_weight [D]`, `attn` (pre-gate [T,Hq,D]), `out` (gated
[T,Hq*D]). Runner `RunDenseAttention` composes gemma-RmsNorm → RopeNeox →
Attention → sigmoid gate and checks both `attn` and `out`.

---

## 6. Task 3 — 35B weight naming + quant scheme (GROUND TRUTH from the real ckpt)

Read 2026-07-03 from `nvidia/Qwen3.6-35B-A3B-NVFP4` snapshot 491c2f1e with our
`dump_container` CLI on all 3 shards + `hf_quant_config.json`. **This CORRECTS
the plan/task assumption that "attn qkv/o are bf16 dense": they are NOT — the
attention AND GDN projections are per-tensor FP8, and lm_head is NVFP4.** The
`hf_quant_config.json` is `MIXED_PRECISION` with three weight classes.

Prefix: every language tensor is under `model.language_model.*`; `lm_head.*` is
top-level. `mtp.*` (19 tensors, shard 3) are EXCLUDED (skip). Shards: layers
0-16 in shard 1, 16-37 in shard 2, 37-39 + `norm` + `lm_head` in shard 3;
`embed_tokens` in shard 1. All quantized weights are torch Linear layout
`[out,in]` → **transpose to Matmul-B `[in,out]` at load**.

**(A) BF16 direct (no dequant):**
| tensor | shape |
|---|---|
| `embed_tokens.weight` | [248320, 2048] (NOT transposed; embed lookup) |
| `norm.weight` (final) | [2048] |
| `layers.{L}.input_layernorm.weight` / `.post_attention_layernorm.weight` | [2048] |
| `layers.{L}.linear_attn.conv1d.weight` | [8192,1,4] → view [8192,4], NOT transposed |
| `layers.{L}.linear_attn.A_log` / `.dt_bias` | [32] (bf16 on disk; upcast→f32 at load, lossless) |
| `layers.{L}.linear_attn.norm.weight` | [128] (RMSNormGated over Dv) |
| `layers.{L}.linear_attn.in_proj_b.weight` / `.in_proj_a.weight` | [32, 2048] → T [2048,32] |
| `layers.{L}.self_attn.q_norm.weight` / `.k_norm.weight` | [256] (full-attn layers) |
| `layers.{L}.mlp.gate.weight` (router) | [256, 2048] → T [2048,256] |
| `layers.{L}.mlp.shared_expert_gate.weight` | [1, 2048] → T [2048,1] |

**(B) FP8 per-tensor E4M3** (`weight F8_E4M3 [out,in]` + `weight_scale F32
scalar` + `input_scale F32 scalar` UNUSED). Dequant: `w_bf16 =
bf16(f8_to_f32(weight) * weight_scale)` — reuses `F8E4M3ToF32`, NO group scale,
NO nibble unpack. All transposed to [in,out]:
| tensor | shape [out,in] |
|---|---|
| `layers.{L}.linear_attn.in_proj_qkv.weight` | [8192, 2048] (q 2048 \| k 2048 \| v 4096; key_dim=Hk·Dk=2048, value_dim=Hv·Dv=4096, conv_dim=8192) |
| `layers.{L}.linear_attn.in_proj_z.weight` | [4096, 2048] (z gate = value_dim) |
| `layers.{L}.linear_attn.out_proj.weight` | [2048, 4096] |
| `layers.{L}.self_attn.q_proj.weight` | [8192, 2048] (=2·Hq·Dh, output-gate doubled: 16·256·2) |
| `layers.{L}.self_attn.k_proj.weight` / `.v_proj.weight` | [512, 2048] (=Hkv·Dh=2·256) |
| `layers.{L}.self_attn.o_proj.weight` | [2048, 4096] (in=Hq·Dh=4096) |

**(C) NVFP4 W4A16 modelopt, group 16** (`weight U8 [out,in/2]` + `weight_scale
F8_E4M3 [out,in/16]` + `weight_scale_2 F32 scalar` + `input_scale F32` UNUSED).
Dequant = `DequantNvfp4ToBf16` (moe-semantics §8). All transposed to [in,out]:
| tensor | weight U8 shape | in_dim |
|---|---|---|
| `layers.{L}.mlp.experts.{0..255}.gate_proj.weight` / `.up_proj.weight` | [512, 1024] | 2048 |
| `layers.{L}.mlp.experts.{0..255}.down_proj.weight` | [2048, 256] | 512 |
| `layers.{L}.mlp.shared_expert.{gate,up,down}_proj.weight` | same as experts | |
| `lm_head.weight` | [248320, 1024] | 2048 → T [2048,248320] |

gate_proj+up_proj of one expert SHARE one `weight_scale_2`; irrelevant to
per-tensor dequant.

**GDN in_proj fusion decision (§1c):** the loader KEEPS the four GDN input
projections SEPARATE (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`),
each dequant+transposed independently. The pinned `in_proj_qkvz`/`in_proj_ba`
fusion is an upstream SHARDED-loading naming convenience (concat along the
output dim); at TP=1 applying each projection as its own matmul is numerically
identical and simpler for owned-memory + the expert-gather loop. Ditto
`qkv_proj`/`gate_up_proj`: q/k/v and gate/up are kept separate. §1c explicitly
permits this ("or keep them separate and adapt the GDN op call").

**Real-tensor spot values** (snapshot 491c2f1e, bf16 bit patterns, baked into
`tests/vllm/test_qwen36_weights.cpp`): NVFP4 `layers.0.experts.0.down_proj`
dq[0,0]=0x3C7E, dq[5,17]=0x3BCB; FP8 `layers.3.self_attn.q_proj` scale
7.6294e-4, dq[0,0]=0x3C0A (f8 0x53), dq[1,5]=0x3CE1 (f8 0x61); router
`layers.0.mlp.gate` [256,2048] dq[0,0]=0x3BB3.

**Memory model:** the loader materializes every weight into OWNED host bf16
buffers (A_log/dt_bias as f32) in Matmul-B layout, so the mmap'd
`SafetensorsFile`s can be released after load. Full 35B bf16 ≈ 70 GB host — fine
on GB10 unified for M0.9 single-seq correctness; device placement is M1.
