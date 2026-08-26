# DeepSeek-V4-Flash — MTP (Multi-Token Prediction) self-speculative decode

**Claim:** `CLAIM-DEEPSEEK-V4-MTP`. **Row:**
`MODEL-SPEC-deepseek-v4-deep-seek-v4-mtp` (`DeepSeekV4MTPModel`, promoted SPIKE→W1 here).
**State:** W1 — loader + tiny-shape draft-forward + lossless-verify unit gate landed on CPU.
The real-model gate is **weight-BLOCKED** (see §4). Runner propose/verify integration is a
NAMED residual (§5).

**Base:** branch `deepseek-v4-mtp` off `main` HEAD `aed4a498`.
**Pinned oracle:** vLLM `5559679229` (0.26.0.dev0) on `dgx.casa`.

---

## 0. Headline

DeepSeek-V4-Flash ships a native **MTP head** (`num_nextn_predict_layers = 1`): one extra
decoder layer that predicts the *next* token from the current post-layer hidden state, so the
engine can **self-speculate** — the draft head proposes k=1 token/step, the full model verifies
by rejection sampling. MTP is **LOSSLESS by construction**: the target verifies every draft
token, so MTP-on greedy output is **token-IDENTICAL** to MTP-off. The win is throughput
(~1.5-2x tokens/step when acceptance is decent), NOT quality.

This is a **WIRING** job: the speculator framework already exists for Qwen3.6-27B/35B
(`src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp`, `rejection_sampler.cpp`). This lane
connects DeepSeek-V4's native MTP head to it.

**Two hard realities discovered while grounding (both material):**

1. **The DS4 forward is NOT the Qwen paged-runner path.** The existing MTP speculator
   (`MtpProposePrefill`) is hard-typed to `Qwen3_5MTPModel` and driven from the GPU paged
   model-runner (`runner.cpp`) over GDN/paged-attention state. DeepSeek-V4 runs on its OWN
   composition path (`deepseek_v4.cpp::ForwardComposeImpl` / `ForwardResidentDecodeGguf` with
   `DeepseekV4KvCache`, a keep-quant CPU-orchestrated + device-resident decode). So the DS4 MTP
   is a **DS4-native** propose/verify built on the DS4 forward + the shared `RejectionSampler`,
   NOT a direct reuse of `MtpProposePrefill`. (The rejection sampler IS shared verbatim.)

2. **The MTP head weights are ABSENT from every available GGUF** (§4). The real-model gate is
   therefore blocked on a re-conversion that retains the `mtp.*` tail.

---

## Scope

Wire DeepSeek-V4's native MTP (`num_nextn_predict_layers=1`) self-speculative draft head to the
existing spec-decode framework: load the nextn weights, build the draft forward, verify with the
shared greedy `RejectionSampler` (lossless — MTP-on greedy == MTP-off greedy). W1 lands the loader
+ tiny-config host draft forward + the lossless gate; the decode-loop + engine registration are
residuals; the real-model gate is weight-blocked (§4).

## Upstream chain

- `vllm/models/deepseek_v4/nvidia/mtp.py:72-390` — `DeepSeekV4MTP` / `DeepSeekV4MultiTokenPredictor`
  / `DeepSeekV4MultiTokenPredictorLayer` (the V4-specific MTP; registry.py:617).
- `vllm/model_executor/models/deepseek_mtp.py` — `SharedHead` + the generic V3 MTP (contrast).
- Pinned oracle vLLM `5559679229` (0.26.0.dev0) on `dgx.casa`.

## Our baseline

- Branch `deepseek-v4-mtp` off `main` `aed4a498`.
- Landed DS4 model (`deepseek_v4.{h,cpp}`, `deepseek_v4_weights.cpp`) + the shared
  `RejectionSampler` (`src/vllm/v1/spec_decode/rejection_sampler.cpp`) from the Qwen3.6 MTP lane.

## Port map

| Ours | Upstream |
|---|---|
| `DeepseekV4MtpDraftLogitsHost` (deepseek_v4.cpp) | nvidia/mtp.py:128-170 forward + :231-258 compute_logits |
| `DeepseekV4MtpHostWeights` (deepseek_v4.h) | nvidia/mtp.py:80-125 (enorm/hnorm/e_proj/h_proj/hc_head/shared_head/mtp_block) |
| `DeepseekV4GgufHasMtp` (deepseek_v4_weights.cpp) | deepseek_mtp.py missing-MTP-layer `ValueError` |
| `DeepseekV4TargetMtpResidualHost` (deepseek_v4.cpp) | nvidia/mtp.py:139-141 previous_hidden_states.view(-1,hc,H) |
| shared `RejectionSampler` | rejection_sampler_utils.py::rejection_sample (verbatim) |

## Tests to port

| Test | Covers |
|---|---|
| `tests/vllm/models/test_deepseek_v4_mtp.cpp` | draft forward finiteness/determinism, RED-first miswires, the lossless self-spec equivalence, the GGUF absence fact |

---

## 1. What the DS4 MTP head IS (ground truth, file:line on both sides)

**Upstream (pinned vLLM, the V4-specific module — NOT the generic `deepseek_mtp.py`):**
`vllm/models/deepseek_v4/nvidia/mtp.py` — `DeepSeekV4MultiTokenPredictorLayer` /
`DeepSeekV4MultiTokenPredictor` / `DeepSeekV4MTP` (registry.py:617
`"DeepSeekV4MTPModel": ("vllm.models.deepseek_v4", "DeepSeekV4MTP")`).

The V4 MTP layer (nvidia/mtp.py:69-176) differs from V3's `eh_proj` (deepseek_mtp.py):
- `enorm`, `hnorm` : RMSNorm[H] over the embed and the previous hidden.
- **separate** `e_proj`, `h_proj` : ReplicatedLinear[H→H] (fp8-quant), *not* a fused
  `eh_proj[2H→H]`.
- `hc_head_fn[hc, hc*H]`, `hc_head_base[hc]`, `hc_head_scale[1]` : the MHC head-collapse params
  (its OWN copy, mirroring the main model's `hc_head_*`).
- `shared_head = { norm[H], head=lm_head[V,H] }` : its OWN norm + lm_head (checkpoint-owned).
- `mtp_block = DeepseekV4DecoderLayer` : a **full V4 decoder layer WITH MHC** (attention + MoE +
  the hyper-connection residual manifold), not a plain transformer block.
- `embed_tokens` : SHARED with the target (checkpoint tensor `mtp.0.emb.tok_emb.weight` maps to
  the shared embed; V4 also carries its own copy in the tail).

**Forward** (nvidia/mtp.py:128-170), previous_hidden = the target's **pre-hc_head MHC residual
stream** flat `[T, hc*H]` (NOT the post-final-norm `[T,H]` hidden — this is the key V4 twist):
1. `inputs_embeds = embed(input_ids)`; **mask position 0 → 0** (pos-0 needs no MTP).
2. `inputs_embeds = enorm(inputs_embeds)`; `prev = hnorm(prev.view(T,hc,H))`.
3. `hidden[T,hc,H] = h_proj(prev) + e_proj(inputs_embeds).unsqueeze(-2)` (e broadcast over hc).
4. `hidden, residual, post_mix, res_mix = mtp_block(positions, x=hidden)` — one V4 decoder layer.
5. `hidden = mhc_post(hidden, residual, post_mix, res_mix)`; return `hidden.flatten(1)` = the
   next-step pre-hc_head residual `[T, hc*H]`.
6. `compute_logits` (nvidia/mtp.py:231-258): `hc_head_collapse(hidden) → shared_head.norm →
   lm_head` → draft logits `[T,V]`.

**Checkpoint tensor names** (nvidia/mtp.py:294-390 `load_weights`): `mtp.{i}.*` remapped to
`model.layers.{num_hidden_layers+i}.*`. Shared-head remap: `.emb.tok_emb→.embed_tokens`,
`.head→.shared_head.head`, `.norm→.shared_head.norm`; everything else →`.mtp_block.*`.

**Ours (reused verbatim):**
- MHC primitives — `include/vllm/model_executor/models/deepseek_v4_mhc.h`:
  `MhcPre` (:132), `MhcPost` (:158), `HcHeadCollapse` (:170).
- Decoder-layer interleave — `deepseek_v4.cpp::ForwardComposeImpl:1649-1735` (attn MhcPre →
  `AttentionBlock:634` → ffn MhcPost+MhcPre → `MoeBlock:881` → final MhcPost). The MTP draft
  forward mirrors ONE iteration of this loop with the eh-lift as its input residual stream.
- `RmsNorm:566`, `Gemm:410` (the shared linear/lm_head).
- Rejection verify — `src/vllm/v1/spec_decode/rejection_sampler.cpp::forward:52`
  (`vt::GreedyRejectionSample`) — the SHARED lossless greedy accept rule.

---

## 2. Config + tensor facts

- `num_nextn_predict_layers = 1` (GGUF KV `deepseek4.nextn_predict_layers`, parsed at
  `deepseek_v4_weights.cpp:498`; safetensors `ParseDeepseekV4Params`, `deepseek_v4_weights.cpp`). Single draft layer.
- The MTP block is layer index `num_hidden_layers` (=43). `p.compress_ratio(43)` returns 0 (out
  of the `[0,43)` range → `DeepseekV4Params::compress_ratio:122`), so the MTP block is a **dense**
  MLA layer (no DSA compressor/indexer) and a **learned-gate** MoE (`is_hash_layer(43)` false).
- `compress_ratios` array length is `block_count + nextn = 44` in the real GGUF; the loader keeps
  the `[0,43)` prefix (`deepseek_v4_weights.cpp:563-566`).

---

## 3. What landed (W1)

- **Loader** (`deepseek_v4_weights.cpp`, `deepseek_v4.h`):
  - `DeepseekV4MtpHostWeights` struct — the tiny-config CPU MTP tower (enorm/hnorm/e_proj/h_proj/
    hc_head_*/shared norm+head + the `mtp_block` `DeepseekV4LayerHostWeights`).
  - `DeepseekV4GgufHasMtp(gguf)` — detects whether a GGUF actually carries the nextn tail
    (mirrors vLLM's "checkpoint quantized without MTP layers" guard, nvidia/mtp.py's
    missing-layer `ValueError`). Returns false for both shipped GGUFs (§4).
  - safetensors `mtp.*` accounting un-skipped behind `DeepseekV4MtpWeights` (was silently
    ignored; now counted, matching vLLM's `_rewrite_spec_layer_name`).
- **Draft forward** (`deepseek_v4.cpp::DeepseekV4MtpDraftLogitsHost`): the tiny-shape nextn
  forward + compute_logits, 1:1 with nvidia/mtp.py:128-258, reusing the DS4 composition helpers.
  `V4MtpMiswire` breaks the eh-lift / MHC-head for the RED-first structural gate.
- **Gate** (`tests/vllm/models/test_deepseek_v4_mtp.cpp`):
  - (a) draft forward produces finite logits at tiny shape;
  - (b) a miswire (skip eh_proj / skip hc_head) CHANGES the draft logits (RED-first);
  - (c) **lossless equivalence**: feeding the DS4 draft token + the DS4 target logits through the
    shared `RejectionSampler` yields output token-IDENTICAL to pure target greedy, for BOTH the
    accept case (draft == target argmax → 2 tokens emitted) and the reject case (draft != target
    argmax → 1 corrected token) — the token-identity guarantee.

## 4. BLOCKER — MTP weights absent from the shipped GGUFs

Verified on `dgx.casa` (2026-07-30, llama.cpp `gguf-py`, no GPU):
- `~/w8run/ds4/ds4flash.gguf` (80.7 GB = 86,720,111,488 B): **1328 tensors, blocks 0-42 only,
  ZERO `nextn`/`mtp`/`blk.43` tensors.** KV `deepseek4.nextn_predict_layers = 1` present.
- `~/w8run/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-...-v2-imatrix.gguf`: identical — 1328 tensors,
  blocks 0-42, zero nextn tensors.

The llama.cpp `deepseek4` converter dropped the MTP layer during GGUF conversion (the metadata
advertises `nextn_predict_layers=1` but no tensors back it). This is exactly vLLM's documented
failure case (deepseek_mtp.py: *"The checkpoint may have been quantized without including the MTP
layers. Use a checkpoint that includes MTP layer weights, or disable speculative decoding."*).

**Consequence:** the real 80.7 GB single-Spark MTP-on==MTP-off gate + acceptance/speedup
measurement is **blocked on a re-conversion** that retains `blk.43.*`/nextn. `DeepseekV4GgufHasMtp`
returns false → the engine cleanly falls back to MTP-off (no crash, vLLM-parity behavior). The
NVFP4 safetensors checkpoint DOES carry `mtp.*` but is 156.7 GiB (does not fit one GB10).

**ds4-vs-MTP note:** the antirez `ds4` oracle (16.5 tok/s) runs plain autoregressive decode — it
does NOT use the MTP head. So MTP is a **BEAT-ds4 lever**: extra tokens/step our engine can do
that the oracle does not, once a weight-carrying GGUF exists.

## 5. Named residuals (honest stopping point)

- **R1 (weight-blocked):** re-convert a `deepseek4` GGUF retaining `blk.43.*` nextn tensors (or
  run the NVFP4 safetensors on multi-Spark), then the real MTP-on==MTP-off gate +
  acceptance/tok-per-step/wall-clock speedup. Cannot be done with the shipped files.
- **R2:** the DS4-native propose/verify LOOP wired into the decode driver
  (`ForwardResidentDecodeGguf` / the decode graph): stash the target's pre-hc_head residual per
  step, run `DeepseekV4MtpDraftLogitsHost` for the k=1 draft, verify next step via
  `RejectionSampler`. Scoped here; the loop body + the residual-stash plumbing are the next brick.
- **R3:** register `DeepSeekV4MTP` as the speculator for `DeepseekV4ForCausalLM` in the engine
  spec-config path (the C++ analogue of registry.py:617) once R2's driver exists.
- **R4:** device MTP draft forward (reuse the DS4 device kernels) for decode-graph speed; the W1
  landing is the host oracle, mirroring how the DS4 model itself gated host-first.

## Gates

| Gate | Form | Status |
|---|---|---|
| Draft forward finite/deterministic | `test_deepseek_v4_mtp` case 1 at tiny shape | PASS 5/5·29 |
| Each nextn lever load-bearing | RED-first miswires (eh-lift / hc_head / hnorm change the logits) | PASS |
| **Lossless self-spec** | DS4 draft + DS4 target verified by the SHARED `RejectionSampler` == pure target greedy (MTP-on == MTP-off), accept + reject cases | PASS |
| No regression | `test_deepseek_v4_forward` 6/6, `test_deepseek_v4_gguf_load` 12/12 (the `ForwardComposeImpl` residual-capture out-param is inert when null) | PASS |
| Real-model MTP-on==MTP-off + acceptance/speedup | 80.7 GB `ds4flash.gguf` on GB10 | **BLOCKED** — no nextn tensors in any shipped GGUF (§4) |

## Dependencies

- The DS4 target forward + host composition (`ForwardComposeImpl`, `AttentionBlock`, `MoeBlock`,
  MHC `MhcPre`/`MhcPost`/`HcHeadCollapse`) — LANDED (DS4 campaign W3-W8).
- The shared greedy `RejectionSampler` (`src/vllm/v1/spec_decode/rejection_sampler.cpp`) — LANDED
  (Qwen3.6 MTP). Reused verbatim; the DS4 lane adds NO new verify path.
- A `deepseek4` GGUF that RETAINS the `blk.{block_count}.*` nextn tail — DOES NOT EXIST yet
  (both shipped files dropped it); blocks the real gate (R1).

## Work breakdown

| Brick | Scope | State |
|---|---|---|
| W1a loader | `DeepseekV4MtpHostWeights` struct + `DeepseekV4GgufHasMtp` absence guard + un-skip the safetensors `mtp.*` accounting | DONE |
| W1b draft forward | `DeepseekV4MtpDraftLogitsHost` (nextn layer + compute_logits, 1:1 nvidia/mtp.py) + `DeepseekV4TargetMtpResidualHost` residual stash | DONE |
| W1c gate | `test_deepseek_v4_mtp` (finite + RED-first + lossless verify) | DONE |
| R2 decode-loop | DS4-native propose/verify over `ForwardResidentDecodeGguf` (stash residual, draft k=1, verify) | RESIDUAL |
| R3 engine register | wire `DeepSeekV4MTP` as the engine speculator (C++ analogue of registry.py:617) | RESIDUAL |
| R4 device draft | device MTP forward for decode-graph speed | RESIDUAL |

## Risks / decisions

- **DECISION:** the DS4 MTP is NOT driven by `MtpProposePrefill` (Qwen-typed, paged-runner-bound);
  only the `RejectionSampler` is shared. The DS4 forward is a separate composition path, so the
  propose/verify loop is DS4-native (R2). Recorded as a correction to the W0 model-matrix note.
- **RISK (materialized):** the shipped GGUFs advertise `nextn_predict_layers=1` but carry no nextn
  tensors → the real gate is weight-blocked. Mitigation: `DeepseekV4GgufHasMtp` returns false and
  the engine falls back to MTP-off (vLLM-parity), so nothing crashes; the win is simply unavailable
  until a weight-carrying GGUF exists.
- **DECISION:** gate host-first at tiny synthetic shape (the real 167B does not fit one GB10),
  mirroring how the DS4 model itself was gated. Device MTP forward is R4.
