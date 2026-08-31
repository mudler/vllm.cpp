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

**CORRECTION (2026-08-31) — THE SPEC AND THE LOADER DISAGREED, AND THE LOADER WAS RIGHT.**
The blocker above is true of the two GGUFs and of the 156.7 GiB NVFP4 safetensors. It is
NOT true of the checkpoint this project already has on the NAS, and
`deepseek_v4_weights.cpp` has said so in a comment since 2026-08-24 while this section
still read `BLOCKED`.

`/mnt/nas_share/rc/ckpt/dsv4-flash-0731-spark-exl3` (100 GB, tp4 ranks) carries
**THREE** MTP layers -- `mtp.{0,1,2}.*`, 3985 keys across
`carried-00{3,4,5}.safetensors`, covering `mtp.L.attn.*`, `mtp.L.attn_norm.weight` and
`mtp.L.ffn.experts.*`. Read from the safetensors headers, the three together are
**8.62 GiB** (~2.87 GiB each): large because each is a full 256-expert MoE layer, but
extractable without the other 91 GB.

**They are NVFP4, not EXL3.** Measured dtypes over the `mtp.*` keys are 1944 `I8` +
1969 `F8_E8M0` + 25 `F8_E4M3` + 20 `BF16` + 27 `F32`, e.g.
`mtp.0.ffn.experts.0.w1.weight` `I8 [2048, 2048]` with `.scale` `F8_E8M0 [2048, 128]`.
That is the config's `packed_e2m1_fp4_with_ue8m0_scales`: only the MAIN model was
requantized to EXL3, and the MTP experts were left in NVFP4. A reader who assumed the
head shares the main model's format would port the wrong dequantizer.

Note also that this checkpoint carries three heads where §0 describes
`num_nextn_predict_layers = 1`. Three is what makes a K5 draft possible upstream, so the
count is a property of THIS artifact and not of the architecture as this spec described it.

**Our loader already parses these keys and discards them deliberately**, matching
`name.rfind("mtp.", 0) == 0` and counting `skipped_mtp_tensors`, which mirrors vLLM's
`AutoWeightsLoader(skip_substrs=["mtp."])` (nvidia/model.py:1474). Upstream skips them in
the MAIN loader precisely because a SEPARATE model owns them (`DeepSeekV4MTP`,
registry.py:617) -- this spec's own R3.

So **R1 is not blocked on a re-conversion.** It is an un-skip of tensors the loader
already reads, plus an NVFP4 expert dequant path for the head, then R2 and R3. What
remains genuinely unmeasured is stated as such: the checkpoint is tp4, and whether one
GB10 holds the arm it belongs to has not been measured. 8.62 GiB is a byte count from the
headers, not a residency figure.

**ds4-vs-MTP note:** the antirez `ds4` oracle (16.5 tok/s) runs plain autoregressive decode — it
does NOT use the MTP head. So MTP is a **BEAT-ds4 lever**: extra tokens/step our engine can do
that the oracle does not, once a weight-carrying GGUF exists.

## 5. Named residuals (honest stopping point)

- **R1 (NO LONGER weight-blocked; see the §4 correction):** the NAS EXL3 checkpoint carries
  three `mtp.{0,1,2}.*` heads (8.62 GiB together, NVFP4 not EXL3) that our loader already
  parses and skips, so this is an un-skip plus an NVFP4 expert dequant, not a re-conversion. Then the real MTP-on==MTP-off gate +
  acceptance/tok-per-step/wall-clock speedup. Re-converting a GGUF retaining `blk.43.*`
  remains an option for the GGUF arm specifically, not a precondition for the gate.
- **R2:** the DS4-native propose/verify LOOP wired into the decode driver
  (`ForwardResidentDecodeGguf` / the decode graph): stash the target's pre-hc_head residual per
  step, run `DeepseekV4MtpDraftLogitsHost` for the k=1 draft, verify next step via
  `RejectionSampler`. Scoped here; the loop body + the residual-stash plumbing are the next brick.
- **R3:** register `DeepSeekV4MTP` as the speculator for `DeepseekV4ForCausalLM` in the engine
  spec-config path (the C++ analogue of registry.py:617) once R2's driver exists.
- **R4:** device MTP draft forward (reuse the DS4 device kernels) for decode-graph speed; the W1
  landing is the host oracle, mirroring how the DS4 model itself gated host-first. This is not
  optional for the throughput claim: `num_experts_per_tok` is 6 of 216, so one token's routed
  experts are 6 x 3 x [2048, 4096] = 604 MB of f32 weights, and a host draft forward cannot be
  the fast path at any batch size. R1b's borrowed views exist so the head can stay quantized
  until a device path consumes it.

## Gates

| Gate | Form | Status |
|---|---|---|
| Draft forward finite/deterministic | `test_deepseek_v4_mtp` case 1 at tiny shape | PASS 5/5·29 |
| Each nextn lever load-bearing | RED-first miswires (eh-lift / hc_head / hnorm change the logits) | PASS |
| **Lossless self-spec** | DS4 draft + DS4 target verified by the SHARED `RejectionSampler` == pure target greedy (MTP-on == MTP-off), accept + reject cases | PASS |
| No regression | `test_deepseek_v4_forward` 6/6, `test_deepseek_v4_gguf_load` 12/12 (the `ForwardComposeImpl` residual-capture out-param is inert when null) | PASS |
| Real-model MTP-on==MTP-off + acceptance/speedup | 80.7 GB `ds4flash.gguf` on GB10 | **BLOCKED for the GGUF arm** — that file has no nextn tensors (§4) |
| Real-model MTP-on==MTP-off + acceptance/speedup | NAS `dsv4-flash-0731-spark-exl3`, `mtp.{0,1,2}.*` un-skipped | **OPEN, not blocked** — three NVFP4 heads are present (§4 correction); needs R1 un-skip + NVFP4 expert dequant + R2/R3 |

### WHICH ARM — the two the goal needs are not the same one

Measured from the safetensors headers of the NAS EXL3 checkpoint (177 files,
2026-08-31), because this decides where every later brick lands:

| part | GiB |
|---|---|
| routed experts | 82.59 |
| everything else | 8.23 |
| **resident, tp1, mtp skipped** | **90.82** |
| MTP heads (all three) | 8.62 |

A GB10's ~119 GiB usable pool leaves **~28 GiB of headroom**, so all three heads
load with roughly 19.6 GiB left for KV and activations. The loader already
coalesces TP4 to TP1, so the four rank files are disjoint slices and this is the
whole tower rather than a quarter of it.

That matters because the two arms split the two properties this row needs. The
**GGUF** arm runs on one Spark at ~14.96 tok/s and carries NO MTP tensors, the
converter having dropped them. The **EXL3** arm carries three heads and, by the
number above, also fits. So the MTP gate belongs on the EXL3 arm, and the GGUF
nextn re-conversion is one way to reach the other arm rather than a precondition
for the row.

§0's "MEMORY-INFEASIBLE on one GB10" is NOT a verdict on this checkpoint. It is
about the 156.7 GiB NVFP4 safetensors. Two artifacts, two verdicts, and this
spec previously named only one of them.

Still unmeasured, and named so it is not mistaken for settled: whether the
carried FP8 half is widened at load, which would move the 8.23 GiB row, and the
real resident total from `ReportDeepseekV4Exl3Residency` on the box. 90.82 GiB is
a packed byte count taken from headers, not a residency measurement.

### THE ARTIFACT'S HEAD IS NOT THE SHAPE W1b BUILT FOR

W1b implemented the draft forward against vLLM's `nvidia/mtp.py`, whose checkpoint
carries `enorm`, `hnorm`, `e_proj`, `h_proj`, `shared_head` and `hc_head_*`
(`_rewrite_spec_layer_name`, :504-508). `DeepseekV4MtpHostWeights` mirrors exactly
that.

The SparkInfer artifact this row would gate on carries something else. Its 32
non-expert tensors per head are, in full: `attn.{attn_sink, kv_norm.weight,
q_norm.weight, wq_a, wq_b, wkv, wo_a, wo_b}`, `attn_norm.weight`,
`ffn.gate.{weight, bias}`, `ffn.shared_experts.w{1,2,3}`, `ffn_norm.weight`,
`hc_attn_{fn, base, scale}`, `hc_ffn_{fn, base, scale}`, and --
**`main_norm.weight` and `main_proj.weight`**. There is no `enorm`, no `hnorm`,
no `e_proj`, no `h_proj`, no `shared_head`, and no `hc_head_*`.

`main_proj.weight` is `F8_E4M3 [4096, 12288]`, i.e. `[H, 3H]`. That is not
DeepSeek-V3's fused `eh_proj`, which is `[H, 2H]` over
`concat(enorm(embed), hnorm(hidden))`. A THIRD hidden-width input goes into this
projection and this spec does not yet know what it is. `hc_attn_fn` is
`[24, 16384]`, so `hc_mult = 4` and `(2 + hc) * hc = 24` -- the head carries its
own MHC mixing, per-block rather than the single `hc_head` collapse our struct
holds.

**RESOLVED (2026-08-31), by reading the producer.** `exllamav3` at the registered
pin `2398c056` carries the architecture in
`exllamav3/architecture/deepseek_v4_mtp.py` and
`exllamav3/modules/arch_specific/dspark.py`, and it is not a DeepSeek MTP head at
all. It is the **DSPARK BLOCK DRAFTER**:

- `main_proj` is `Linear(in_features = n_taps * hidden_size, out_features =
  hidden_size)`. `dspark_target_layer_ids = [40, 41, 42]`, so `n_taps = 3` and the
  `[H, 3H]` shape is the concatenation of the TRUNK'S STREAM-MEAN TAPS at those
  three layers -- not `concat(embed, hidden)`. `main_norm` normalizes the result.
- The three `mtp.{0,1,2}` are **three blocks of ONE drafter**, not three
  independent heads: full transformer blocks with compressor-less DSA attention, a
  216-expert noaux MoE with a shared expert, and mHC residual streams. Entry is
  `main_proj`/`main_norm` on block 0; exit is block N-1's own `hc_head` collapse
  and final norm into the SHARED TRUNK HEAD, which is why no `shared_head` or
  `lm_head` appears in the tail.
- It is SEMI-AUTOREGRESSIVE. `dspark_block_size = 5` is the K5 in "DSpark K5":
  the input layer takes one seed token, appends `block_size - 1` copies of
  `dspark_noise_token_id` (128799), embeds them through the ATTACHED target's
  embedding, and expands to the mHC stream stack. It proposes a BLOCK, and the
  published acceptance profile 0.65/0.44/0.31/0.17/0.07 is one number per block
  position.
- The last block additionally carries a factorized-bigram markov head
  (`dspark_markov_rank = 256`, a per-token logit bias applied during the
  sequential sampling loop) and a confidence head (a per-position score for
  dynamic draft length). Neither has any analogue in this tree.

`num_nextn_predict_layers = 1` in the artifact's own `config.json` does NOT
describe this. The block count is `num_mtp_layers`, and three blocks are what the
tensors show.

A reference implementation ships in the source repo under
`hf/inference/model.py` (`DSparkBlock`, `DSparkAttention`, `forward_spec`). It is
NOT in the staged copy on the NAS, so reading it needs the HF repo.

**Nothing here should be mapped onto our struct by name similarity.** Guessing
that `main_proj` is `eh_proj` with an extra stream would produce a head that runs,
emits finite logits, and drafts badly -- and because MTP is lossless by
construction, the OUTPUT would still be correct, so the only symptom would be an
acceptance rate nobody can explain. That is the most expensive shape of wrong
available here.

What settled it was the producer rather than inference. Note that `787d1582`, the
rev the checkpoint was quantized at, does NOT contain `main_proj`; the DSpark
support is at the registered pin `2398c056`, so a reader who checks only the
quantizing rev finds nothing and may conclude the name is undocumented.

### R1b — why there is no host float tower for the head

Arithmetic first, because it removes an option rather than choosing one. One
head's routed experts are 216 experts x 3 projections x [2048, 4096] = 5.44G
values. As host f32 that is **20.2 GiB per head and 60.8 GiB for the three this
artifact carries**, next to a target that already fills the box. So
`DeepseekV4MtpHostWeights`, which is all `std::vector<float>`, cannot hold this
head, and R1b does not try.

The tail is BORROWED instead. `RouteDeepseekV4MtpTail` builds per-head views that
point into the shard mapping and own nothing, and
`DequantizeDeepseekV4MtpTensor` expands exactly one tensor when a caller wants it,
choosing the reader from the classified format so a consumer cannot pick the wrong
one.

One trap is written into the type. For MXFP4 the stored shape is `[N, K/2]`,
because two e2m1 nibbles share a byte, so the view carries a LOGICAL `in_dim` of
`2 * shape[1]`. Reading `shape[1]` as the width halves every routed expert and
still produces finite numbers, which no shape assertion would catch; the gate
mutates exactly that.

The gate hand-computes its expectations from the formats rather than comparing
against the same helper the code calls -- otherwise it would prove the two agree
and still pass if the wrong reader were chosen. It also carries a SECOND fp8
block row with a different scale, because the first version read only row 0, where
every block extent picks `scale[0]`, and a mutation changing the extent from 128
to 64 survived it.

### R1a — what landed, and what it deliberately does NOT do

`ClassifyDeepseekV4MtpTail` turns the blanket skip into an accounted inventory:
per-format counts, a head count taken from the TENSORS rather than from
`num_nextn_predict_layers`, and a by-name report for any layout this arm has no
reader for. The loader fills it on every safetensors load
(`w.exl3.mtp`), and the skip itself is unchanged -- vLLM skips the tail in the
main loader too, because a separate model owns it.

It is a REPORT, never a throw. The 156.7 GiB NVFP4 checkpoint's tail uses the
double-scale variant (`weight_scale` + `weight_scale_2` + `input_scale`), and
throwing here would break loads that work today. R1b refuses at the point where a
head is actually wanted.

Gated two ways, because the two claims are different. `test_deepseek_v4_mtp_inventory`
states the measured artifact's real shapes as data and exercises the classifier
directly: group-16 NVFP4 is refused rather than read as MXFP4, a quantized weight
with no scale is refused by name, and a scale that does not tile its weight at
128x128 is refused. `test_deepseek_v4_exl3_loader` drives the PRODUCTION load and
reads the inventory off the weights, so deleting the loader's call site reds it --
which is what separates "the function works" from "anything reaches it".

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
| R1a inventory | the loader CLASSIFIES the `mtp.*` tail instead of only counting it (`ClassifyDeepseekV4MtpTail`); fp8-block at 128x128 and MXFP4 at group 32 are recognised, anything else is reported by name | DONE |
| R1b routing | the tail is routed to BORROWED views (`RouteDeepseekV4MtpTail`) and dequantized one tensor at a time (`DequantizeDeepseekV4MtpTensor`); the residency question is answered by keeping it quantized, NOT by a host float tower | DONE |
| R1c drafter model | the tail is a DSPARK BLOCK DRAFTER, not a DeepSeek MTP head: 3 blocks, 3 trunk taps at layers [40,41,42] through `main_proj`, a 5-token noise-seeded block, a markov bias head and a confidence head. `DeepseekV4MtpHostWeights` and `DeepseekV4MtpDraftLogitsHost` model a DIFFERENT architecture and cannot be extended into this one | RESIDUAL, and it is a NEW model rather than a wiring job |
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
