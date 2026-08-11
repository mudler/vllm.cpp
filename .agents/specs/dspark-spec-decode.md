# DSpark speculative decoding — spike spec (`SPEC-DSPARK`)

| Field | Value |
|---|---|
| Row | `SPEC-DSPARK` (engine-matrix), feature-matrix §8 "DSpark (semi-autoregressive block drafter)" |
| Scope | Port DSpark semi-autoregressive block speculative decoding for the Qwen3 and Gemma4 draft families onto the landed `SPEC-DFLASH` lane: config/method resolution, the Markov logit-bias head and draft model, native + Speculators checkpoint loading, the anchor-as-first-prediction query layout, sequential Markov draft sampling (greedy then probabilistic), runner/one-surface wiring, and the token-exact + acceptance + speed gates. **Excluded:** DeepSeek-V4 DSpark (`models/deepseek_v4/*/dspark.py`, HW-blocked), the confidence head (upstream does not wire it), TLI heterogeneous-vocabulary drafting (`SPEC-TLI`), and DSV4 sparse-MLA noncausal attention. |
| Upstream chain | `config/speculative.py:62,310,706-709,869-887,934-964,1004-1027,1333` → `v1/core/sched/scheduler.py:261-265` → `v1/worker/gpu/spec_decode/__init__.py:17-18` → `v1/worker/gpu/spec_decode/dspark/speculator.py:37,76,100,151` → `dspark/utils.py:15` → `model_executor/models/qwen3_dspark.py:36,70,95,132-147,149-185` (+ `gemma4_dspark.py:134,182`) → `transformers_utils/configs/speculators/algos.py:120-165`, all @ `555967922` |
| Roadmap | `ROAD-V1-C3` spec-decode named tail (with `SPEC-TLI`) |
| Role / claim | helper, branch `row/SPEC-DSPARK` |
| Base | `bc6e3d7216523c40fca75c47fec7c5777d04d64c` (origin/main, 2026-08-09) |
| Parity pin | vLLM `555967922` (0.26.0.dev0) at `$VLLM_SOURCE` |
| Supersedes | [dspark-speculator-note.md](dspark-speculator-note.md) — the 5-line 2026-08-08 grounding rider (kept; this is its promised full scope) |
| Our baseline | The landed `SPEC-DFLASH` lane (`CLAIM-DFLASH-D14`, `DONE`): `include/vllm/v1/worker/gpu/spec_decode/dflash/speculator.h` + `src/.../dflash/speculator.cpp`, `include/vllm/model_executor/models/qwen3_dflash.h` (+ `_weights.cpp`, `_gguf.cpp`), the runner branch `src/vllm/v1/worker/gpu/runner.cpp:1183,1760-1761,1874-1891`, `include/vllm/config/speculative.h` (`method` already accepts the string `"dspark"` in `use_eagle()` `:203-204`; `PrepareDflashInputs.sample_from_anchor` exists at `qwen3_dflash.h:320`, always `false`). Detail in §2. |
| Port map | §2 (the six-item A–F delta table: Markov head, sequential sampling, anchor layout, `d2t` reduced vocab, config resolution, Speculators translation) and §4 (which file each lands in). |
| Tests to port | §5 — `tests/v1/e2e/spec_decode/test_spec_decode.py:1489-1530` (`dspark_config` + `test_dspark_correctness_and_acceptance_rate`) and `:291-350` (Gemma4), `tests/models/registry.py:1467-1476`; `tests/v1/attention/test_dspark_noncausal_sparse_mla.py` checked in SKIPPED (DSV4 sparse MLA, out of scope). |
| Gates | §5 — token-exact (or ratified near-tie) vs the pinned oracle on the same target+draft+k, spec-OFF byte-identical, acceptance rate/length at or above the upstream band, our DSpark-ON at or above vLLM DSpark-ON on every throughput axis, and `scripts/check-surface-coverage.py` green. |
| Dependencies | Landed: `SPEC-DFLASH` (`DONE`), `SPEC-REJECTION` verify half, `SPEC-GDN-SEGMENTS`. External, PENDING developer authority: checkpoint downloads (2.79-8.80 GB), dgx GPU time, push/draft-PR. Blocking unknown: R1, whether the pinned oracle runs DSpark at all. |
| Work breakdown | §4 — W1 config, W2 Markov head + draft model, W3 loader (native + Speculators), W4 speculator (anchor layout + sequential sampling), W5 runner + one-surface, W6 gates. W1-W4 are CPU-gateable. |
| Risks/decisions | §6 — R1 oracle runnability (V2 runner), R2 Speculators format is a new subsystem, R3 the community 27B checkpoint's `attn_output_gate`, R4 the `k >= dspark_block_size` garbling trap, R5 sequential sampling vs CUDA-graph capture, R6 greedy before probabilistic, R7 GB10 host-RAM pressure. |
| Status | W1-W5 LANDED and DSpark now genuinely speculates on the 35B gate model (real acceptance, 6.78 -> 41.89 tok/s) after fixing an engine-wide `check_for_draft_tokens` wiring bug that silently disabled EVERY speculator on the CLI/server path. W6 PARTIAL: spec-ON output is token-identical to spec-OFF and reproducible on the 35B gate model (§6b), but speed is ~2% BEHIND spec-off and the cross-engine + acceptance-band gates are still owed. R1 answered (§6a). |
| Goal (developer, 2026-08-09) | a FULL DSpark implementation in vllm.cpp, mirrored from vLLM |

## 0. Verdict

DSpark is reachable on top of the landed DFlash lane, and it is a **small
additive delta, not a new mechanism**. Upstream's entire DSpark surface is
**1613 lines across 5 files**, and 3 of those files are `class X(DFlashY)`
subclasses:

| Upstream file @ `555967922` | lines | relation to what we already ship |
|---|---:|---|
| `v1/worker/gpu/spec_decode/dspark/speculator.py` | 169 | `DSparkSpeculator(DFlashSpeculator)` — 2 overrides |
| `v1/worker/gpu/spec_decode/dspark/utils.py` | 74 | draft build/alias, ~= our `LoadQwen3DFlash` seam |
| `model_executor/models/qwen3_dspark.py` | 185 | `Qwen3DSparkModel(DFlashQwen3Model)` + Markov head |
| `model_executor/models/gemma4_dspark.py` | 312 | same head over the Gemma4 draft (second target family) |
| `models/deepseek_v4/nvidia/dspark.py` | (DSV4) | **OUT OF SCOPE** — DeepSeek-V4 is HW-blocked (2× Spark) |

Everything DSpark needs that is *hard* — the non-causal in-block attention
primitive, the multi-tap aux feature combine (`fc(cat(aux))`), the context-KV
precompute, the separate-draft-checkpoint loader, the `--speculative-config`
plumbing, the verify/propose runner loop — **is already landed and gated** under
`SPEC-DFLASH` (`CLAIM-DFLASH-D14`, speed gate met). DSpark adds exactly three
things on top (§2).

Draft checkpoints exist for **both gate models** and for a 4B lane that mirrors
the upstream test one-for-one (§3), so this is gateable on-box without the
DeepSeek-V4 hardware blocker.

## 1. What DSpark is (upstream mechanism)

DSpark is **semi-autoregressive block drafting**: DFlash's one-parallel-pass
block draft, plus a cheap sequential pass that re-introduces intra-block
dependency.

Anchors (`555967922`):

1. **Speculator** — `v1/worker/gpu/spec_decode/dspark/speculator.py:37`
   `DSparkSpeculator(DFlashSpeculator)`; overrides `load_draft_model` `:76`,
   adds `_sample_sequential` `:100`, overrides `_generate_draft` `:151`.
   Class docstring `:5-24` states the two differences verbatim.
2. **Draft build** — `dspark/utils.py:15` `load_dspark_model`: same non-causal
   backend resolution as DFlash (`dflash_has_any_non_causal`), same
   embed_tokens / lm_head aliasing to the target (`_should_share`), no PP.
3. **Draft model** — `models/qwen3_dspark.py:36` `DSparkMarkovHead`
   (`markov_w1: [vocab_size, markov_rank]`, `markov_w2: [draft_vocab_size,
   markov_rank]`), `:70` `Qwen3DSparkModel(DFlashQwen3Model)`, `:95`
   `Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM)` with
   `compute_draft_logits` `:132`, `map_draft_to_target` `:137`,
   `markov_embed`/`markov_bias` `:143-147`, `load_weights` `:149-185`.
4. **Config** — `config/speculative.py:62` (`DSparkModelTypes`), `:310`
   (method literal), `:706-709` (target_model_config required; DSV4 ships the
   draft inside the target), `:869-887` (method auto-detect: name contains
   `dspark` OR architectures contain `Qwen3DSparkModel`/`Gemma4DSparkModel`),
   `:934-961` (DSV4 / Gemma4 config normalization), `:963-964`
   (`parallel_drafting = True`), `:1004-1027` (**`num_speculative_tokens >=
   dspark_block_size` or output is garbled, not merely worse**), `:1333`
   `use_dspark()`.
5. **Scheduler lookahead** — `v1/core/sched/scheduler.py:261-265`:
   `num_lookahead_tokens = num_spec_tokens` for DSpark, versus DFlash's
   `num_spec_tokens + 1` `:256-260`. This is the single scheduler-visible
   consequence of anchor-as-first-prediction.
6. **Runner routing** — `v1/worker/gpu/model_runner.py:208` (aux hidden states
   for `eagle3|dflash|dspark`); `spec_decode/__init__.py:17-18` (factory);
   `spec_decode/utils.py:54-75` `get_parallel_drafting_token_id` (mask-token
   resolution order incl. `dspark_noise_token_id`);
   `spec_decode/eagle/eagle3_utils.py:48-50` (`dspark_target_layer_ids` → aux
   layer ids, `+1` indexing).
7. **V2-runner-only** — `config/vllm.py:560-568` forces the V2 runner for
   `method == "dspark"`; `:2168-2177` lists DFlash/DSpark as "parallel drafting
   natively in V2 via their own speculators". (Informational for us: we have one
   runner. See `.agents/vllm-v1-v2.md`.)
8. **Speculators-format translation** — `transformers_utils/configs/speculators/
   algos.py:133-165` `update_dspark`: maps `aux_hidden_state_layer_ids` →
   `eagle_aux_hidden_state_layer_ids` + `target_layer_ids = [i-1]`, sets
   `architectures = ["Qwen3DSparkModel"]`, carries `sample_from_anchor`
   (**default False in this path**), `draft_vocab_size`, `mask_token_id`,
   `markov_rank`, `block_size`, and maps `sliding_window_non_causal` →
   `dflash_config.causal` (`:120-131`, the shared DFlash branch).

### The draft step, precisely

Per request, per decode step:

```
queries      = N = num_speculative_tokens            (sample_from_anchor=True)
             = 1 + N                                 (sample_from_anchor=False, DFlash layout)
q[0]         = anchor  (the last verified/bonus token id)
q[1..]       = noise/mask token id  (mask_token_id | dspark_noise_token_id)

parallel:    head_hidden = DFlash block forward over [context_kv ; queries]
sequential:  prev = anchor_id                        (TARGET vocab)
             for i in 0..N-1:
                 bias_i   = markov_w2( markov_w1[prev] )        # [draft_vocab]
                 logits_i = draft_lm_head(head_hidden[i]) + bias_i
                 tok_i    = argmax(logits_i)          (greedy)  -> map_draft_to_target
                          | gumbel_sample(...)        (probabilistic, target vocab
                                                       after d2t scatter, key pos Q-1)
                 draft[i] = tok_i ; prev = tok_i
```

`sample_pos = query_pos + 1` in the anchor path (standard next-token), against
DFlash's masks-sit-at-the-predicted-position layout; the probabilistic branch
passes `sample_pos - 1` as the Gumbel key because the target verifies a draft
token with its *predecessor's* key (`speculator.py:135-137`).

## 2. Delta versus our landed DFlash lane

Ours today (`SPEC-DFLASH`, `DONE`): `include/vllm/v1/worker/gpu/spec_decode/
dflash/speculator.h` + `src/.../dflash/speculator.cpp` (`DflashProposeBlock`,
`SampleDflashBlockDrafts`), `include/vllm/model_executor/models/qwen3_dflash.h`
(+ `_weights.cpp`, `_gguf.cpp`), runner branch
`src/vllm/v1/worker/gpu/runner.cpp:1183,1760-1761,1874-1891`
(`propose_drafts_dflash`, `set_dflash_draft`, `dflash_tap_layer_ids_`),
`include/vllm/config/speculative.h` (`method` already carries the string
`"dspark"` in `use_eagle()` `:203-204`, and `PrepareDflashInputs` already has a
`sample_from_anchor` field, `qwen3_dflash.h:320`, hardcoded `false`).

| # | New surface | Where it lands | Why it is new |
|---|---|---|---|
| **A** | **Markov head** — `markov_w1 [V, r]` embedding gather + `markov_w2 [V_draft, r]` GEMV, added to the base draft logits | new `qwen3_dspark.{h,cpp}` (+ weights) | no analogue in DFlash; `r = 256` in every shipped ckpt |
| **B** | **Sequential N-step sample loop** with the running `prev` token, replacing DFlash's single parallel argmax | new `spec_decode/dspark/speculator.{h,cpp}` | DFlash samples all block positions independently; DSpark's step *i* depends on step *i−1* |
| **C** | **Anchor-as-first-prediction layout** (`N` queries, `sample_pos = q+1`) | `qwen3_dflash.h` `PrepareDflashInputs.sample_from_anchor` (field already present, currently dead — always `false`) | our only exercised path is `sample_from_anchor=false`. **Already correct, verified**: `NumLookaheadTokens()` (`speculative.h:220-229`) branches on `use_dflash()` (`== "dflash"` only) for `k+1` and falls through to `use_eagle()` — which already lists `"dspark"` — for `k`, matching `scheduler.py:261-265`. No change needed; add the pinning test |
| **D** | **Reduced draft vocab (`d2t`)** — `draft_vocab_size=32000` in both RedHatAI speculators ckpts; greedy remaps ids, probabilistic scatters logits into target columns | draft model + sampler | our DFlash drafts are full-vocab; `draft_id_to_target_id` is unloaded today |
| **E** | **Config**: method `dspark`, `dspark_block_size` floor, `dspark_noise_token_id`, `dspark_target_layer_ids`, `n_predict = block_size` | `include/vllm/config/speculative.h` + `src/vllm/config/speculative.cpp` | method string exists; the resolution rules do not |
| **F** | **Speculators-format config translation** (`algos.py:133`) | new; we have **zero** `speculators` handling today (grep: no hits in `src/`,`include/`) | needed for the RedHatAI gate-model drafts (§3) |

Explicitly **reused unchanged**: the non-causal block attention primitive
(`vt::DFlashBlockAttention`), `CombineAuxFeatures`/`fc`, context-KV precompute,
`ForwardBlockLogitsWithContext`, the multi-tap aux plumbing, the draft-checkpoint
loader shape, the `--speculative-config` JSON parse, the C-ABI/CLI/server
surface, and the verify/rejection loop.

## 3. Checkpoints (HF API, fetched 2026-08-09)

| Checkpoint | Target | Format | Size | Block | `sample_from_anchor` | draft vocab | Why it matters |
|---|---|---|---:|---:|---|---|---|
| `deepseek-ai/dspark_qwen3_4b_block7` | `Qwen/Qwen3-4B-FP8` | native | 2.79 GB | 7 | absent → **True** | full (151936) | **the upstream test's own pair** (`test_dspark_correctness_and_acceptance_rate`, `large_gpu_mark(min_gb=24)`) — smallest honest lane |
| `deepseek-ai/dspark_qwen3_8b_block7` | Qwen3-8B | native | 4.74 GB | 7 | absent → True | full | upstream's default `model` (`speculative.py:875`) |
| `RedHatAI/Qwen3.6-35B-A3B-speculator.dspark` | **our 35B gate model** | speculators | 1.90 GB | 8 | **True** | **32000 + d2t** | binding gate-model lane; 5× SWA layers, hidden 2048 |
| `satgeze/Qwen3.6-27B-DSpark` | **our 27B gate model** | native | 8.80 GB (+3.73 GB GGUF) | 15 | absent → True | full | 27B lane; community ckpt, `attn_output_gate: true` (**risk R3**) |
| `RedHatAI/gemma-4-31B-it-speculator.dspark` | Gemma4-31B | speculators | 8.39 GB | 8 | **False** (1+N layout) | 32000 + d2t | second target family; exercises the DFlash-layout branch |
| `deepseek-ai/dspark_gemma4_12b_block7` | gemma-4-12B-it | native | — | 7 | — | — | the upstream Gemma4 e2e test's pair |

Non-goals: `deepseek-ai/DeepSeek-V4-*-DSpark` (draft ships inside the target;
DSV4 is HW-blocked, see [mla-deepseek-campaign.md](mla-deepseek-campaign.md)),
Kimi-K3/GLM-5.2/MiniMax-M3 DSpark drafts (targets not in scope for this row).

Every shipped config sets `enable_confidence_head: true`, and upstream
**deliberately does not wire it** (`qwen3_dspark.py:165-167`: "confidence_head is
not wired into inference yet; skip its weights"). We mirror that: skip, do not
invent.

## 4. Slice plan

Each slice: red test first, focused gate, full gate, fresh review, then the next.
CPU-runnable through W3; W4+ needs the GPU box.

| Slice | Content | Gate | HW |
|---|---|---|---|
| **W1 config** | accept `"dspark"` in `ParseSpeculativeConfigJson` (+ require the `model` key, as dflash does), `ResolveDspark` (method resolution by name/architectures, `n_predict = block_size`, the `k >= dspark_block_size` hard error), noise-token resolution order, `parallel_drafting`; pin `NumLookaheadTokens() == k` | new `tests/vllm/config/test_speculative_dspark.cpp`. **RED today**: `speculative.cpp:44-49` rejects every method outside `mtp\|dflash\|ngram\|draft_model`, so `{"method":"dspark",…}` throws | CPU |
| **W2 Markov head + draft model** | `Qwen3DSparkModel` = our `Qwen3DFlashModel` backbone + `DSparkMarkovHead`; `compute_draft_logits`, `markov_embed`/`markov_bias`, `map_draft_to_target`, `d2t` | op-level golden vs a dumped HF Markov head (`tools/parity/`), rel-L2 band as in D2 | CPU (+GPU port) |
| **W3 loader** | native key spelling (`model.markov_head.markov_w{1,2}`, `d2t`→`draft_id_to_target_id`, skip `t2d`/`mask_embedding`/`confidence_head`) **and** the speculators translation (`algos.py:133`) incl. `transformer_layer_config` unwrap | loader unit test on the real ckpt tensor list; the D-lane lesson: assert the exact on-disk spelling | CPU |
| **W4 speculator** | anchor layout (`N` queries, `sample_pos=q+1`) + `_sample_sequential`; greedy first, probabilistic (Gumbel + d2t scatter) second | `SampleDsparkBlockDrafts` unit gate on a synthetic fixture (RED: parallel-sample ≠ sequential when the Markov bias is non-zero) | CPU |
| **W5 runner + surface** | `propose_drafts_dspark`, `set_dspark_draft`, aux taps from `dspark_target_layer_ids`, `--speculative-config '{"method":"dspark",...}'` through CLI/server/C-ABI (**POL-ONE-SURFACE**: `include/vllm.h` already carries `speculative_config`; no new ABI symbol expected — confirm with `scripts/check-surface-coverage.py`) | engine e2e; spec-OFF byte-identical (the DFlash inertness discipline) | GPU |
| **W6 gates** | token-exactness vs the pinned oracle running the SAME draft; acceptance rate/length; the on-par-or-above speed A/B vs vLLM `--speculative-config dspark` | §5 | GPU (dgx, flock) |

Order of lanes: **4B first** (upstream's own test pair, 24 GB class, native
format, full vocab, greedy path) → **35B gate model** (speculators format +
reduced vocab + `sample_from_anchor=True`) → **27B** → Gemma4 (`1+N` layout).

## 5. Tests to port (POL-PORT-TESTS) and gates

Upstream test surface at the pin (there is no unit test for the DSpark
speculator; upstream covers it e2e):

| Upstream | Ours |
|---|---|
| `tests/v1/e2e/spec_decode/test_spec_decode.py:1489-1530` `dspark_config` + `test_dspark_correctness_and_acceptance_rate` (Qwen3-4B-FP8 + `dspark_qwen3_4b_block7`, k=7, probabilistic, GSM8K acc ≥ 0.801·0.9, acceptance_rate ≥ 0.428·0.9, acceptance_len ≥ 3.994·0.9) | `tests/parity/test_qwen3_dspark_spec_decode.cpp` (e2e) + the acceptance-rate floor; we additionally gate greedy token-exactness vs the oracle (stricter, per [gates.md](../verification.md)) |
| `tests/v1/e2e/spec_decode/test_spec_decode.py:291-350` `test_gemma4_dspark_correctness_and_acceptance_rate` | deferred to the Gemma4 lane; checked in SKIPPED with the reason until then |
| `tests/models/registry.py:1467-1476` (`Qwen3DSparkModel`, `Gemma4DSparkModel`, `DSparkDraftModel`) | model-registry rows + `docs/FEATURES.md` arch table (CI-bound, 33 archs today) |
| `tests/v1/attention/test_dspark_noncausal_sparse_mla.py` | **OUT OF SCOPE** — DSV4 sparse MLA, already checked in SKIPPED under the MLA campaign |

Binding acceptance for `DONE` (house gate, [verification.md](../verification.md)):

1. **Token-exact** greedy output versus the pinned vLLM oracle running the same
   target+draft+k, and versus our own spec-OFF decode, at c1 — or the ratified
   near-tie distributional form where the oracle's own greedy is non-deterministic.
2. **Spec-OFF byte-identical**: no DSpark code on the non-speculative path.
3. **Acceptance rate/length** at or above the upstream reference band.
4. **Speed**: our DSpark-ON ≥ vLLM's DSpark-ON on every throughput axis, ≤ on
   latency/memory, on an idle box, ≥2 reps, `nsys` both sides with the same tool
   if a kernel claim is made.
5. **One surface**: reachable through `include/vllm.h` + CLI + server, examples
   as thin clients; `scripts/check-surface-coverage.py` green.

## 6. Risks and open questions

- **R1 — oracle runnability.** Before any implementation slice, prove the pinned
  oracle actually *runs* `Qwen/Qwen3-4B-FP8` + `dspark_qwen3_4b_block7` and emits
  a greedy golden (memory: "gateability = model RUNS, not config constructs";
  DSpark forces the **V2 runner**, which our oracle recipes have not exercised).
  If the pinned oracle cannot run DSpark, the whole row is oracle-blocked and
  that is the stop condition, recorded — not worked around.
- **R2 — Speculators format is a new subsystem for us.** Two of the three
  gate-model drafts are speculators-format. Scope it as its own slice (W3) so it
  can be reviewed independently; a `NEEDS_CONTEXT`-style stop is better than
  smuggling it into the loader.
- **R3 — `satgeze/Qwen3.6-27B-DSpark` is a community checkpoint** with
  `attn_output_gate: true`. That key is **not read** by upstream
  `qwen3_dflash.py` / `qwen3_dspark.py` / `qwen3.py` (grep, `555967922`), so
  either it is inert metadata or the tensors are gated and vLLM itself cannot
  load it. Verify against the oracle before adopting it as the 27B lane;
  `Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft` and `Koopah/…-NVFP4-DSPARK` are the
  alternates.
- **R4 — the `k >= dspark_block_size` trap.** Upstream documents that a smaller
  `k` yields *garbled* output, not merely lower acceptance
  (`speculative.py:1004-1027`). Our config gate must be a hard error with a red
  test, not a warning.
- **R5 — sequential sampling versus CUDA graphs.** Upstream captures the whole
  draft step (parallel backbone + the N-step Markov loop) in a FULL graph
  (`speculator.py:22-24`). An N-iteration host loop with a device round-trip per
  step would be a decode-path regression at exactly the point the feature is
  supposed to win. Memory: "CUDA-graph capture bakes stack addresses" — no
  function-local upload temporaries in the captured region.
- **R6 — probabilistic path.** The upstream tests gate at `temperature=1.0`
  with `draft_sample_method: probabilistic` (Gumbel + rejection). Our landed
  spec-decode gates are greedy. Greedy-first is the honest order; the
  probabilistic path is a named, separate slice, not an assumption.
- **R7 — GB10 memory.** `gpu_memory_utilization` reserves HOST RAM on GB10 and a
  big oracle beside ctest has rebooted the box. The 4B lane is chosen partly for
  this; keep the oracle and our engine serialized under the `flock`.

## 6a. R1 RESULT — the oracle RUNS DSpark (2026-08-09)

R1 was the row's one blocking unknown: DSpark forces the V2 runner, which our
oracle recipes had never exercised. **Answered, and the answer is clean.**

Run on dgx, one `flock`, `local-ai-worker` parked, `enforce_eager=True`,
`gpu_memory_utilization=0.30`, two prompts x 48 greedy tokens:

| | value |
|---|---|
| target / draft / k | `Qwen/Qwen3-4B` / `deepseek-ai/dspark_qwen3_4b_block7` / 7 |
| DSpark-ON arm | loads and decodes, exit 0 |
| DSpark-ON vs spec-OFF | **TOKEN-IDENTICAL on both prompts** (48/48 ids each) |
| evidence | `dgx:~/work/dspark-r1/{r1.log,r1_on.json,r1_off.json}` |

**Recorded caveat:** this is the dgx **v0.25.0 stage** oracle
(`~/venvs/vllm-oracle` -> `vllm-oracle-v0.25.0-stage`), NOT the `555967922`
(0.26.0.dev0) pin; `vllm-oracle-next` has no vllm installed, and the pin rebuild
is a standing residual. The 0.25.0 stage does carry the dspark speculator with
`sample_from_anchor`, so the semantics this port mirrors are present, but a
binding W6 gate should be re-run once the pinned oracle exists.

**Environment trap, cost one queued GPU slot:** the first attempt failed at
EngineCore init with `FileNotFoundError: 'ninja'` in BOTH arms — a non-login
tmux shell has neither `~/venvs/vllm-oracle/bin` nor `~/.local/bin` on PATH.
Both arms failing identically reads as "the oracle cannot run this feature",
which is the exact question R1 asks; suspect the environment first.

## 6b. W6 FIRST RUN — runs end to end, but the drafter is INERT (2026-08-10)

Two lanes were run on dgx under one `flock`, worker parked.

**4B lane** (`Qwen/Qwen3-4B` + `deepseek-ai/dspark_qwen3_4b_block7`, k=7).
Our spec-OFF decode reproduces the oracle's golden text exactly. Our DSpark arm
**failed at the first propose**: `missing target aux multi-tap`. Cause: only the
Qwen3.5/3.6 dense + MoE forwards implement `ForwardDeviceMultiTap`, and classic
`Qwen3ForCausalLM` has none — a hole that was LATENT FOR DFLASH TOO. Fixed by
`ModelBase::supports_aux_multi_tap()` + a load-time refusal naming the method
and the architecture (commit `56b7b607`).

**35B gate-model lane** (`nvidia/Qwen3.6-35B-A3B-NVFP4` +
`RedHatAI/Qwen3.6-35B-A3B-speculator.dspark`, k=8, the reduced-vocab `d2t` +
`sample_from_anchor` path):

| arm | result |
|---|---|
| ours spec-OFF | 48 tokens, 37.34 tok/s |
| ours DSpark-ON | 48 tokens, **6.78 tok/s**, text byte-identical to spec-OFF |
| oracle DSpark-ON | runs; its own ON and OFF texts DIVERGE mid-sequence |
| oracle vs ours | our BASE decode already differs from the oracle's on this ad-hoc prompt, so no DSpark-attributable cross-engine claim is possible from it |

**The load-bearing negative results, isolated with `VT_SPEC_TRACE=1`:**

1. **The drafter PROPOSES correctly.** A propose-side trace prints one line per
   step: `rows=1 nqpr=8 drafts/row=8 first=[13 198 12 2972 57590 11 11751 11]` —
   8 plausible target-vocab ids every step, from the anchor-layout block. So W2
   (Markov head), W3 (loader) and W4 (sequential sampler) are doing their job on
   real weights.
2. **The drafts NEVER REACH THE VERIFY PATH.** Zero `[SPECTRACE]` lines (the
   verify side, `kr > 0`) and zero `[spec-install]` lines. Precisely: the
   `[spec-install]` probe sits AFTER the two `continue`s in
   `Scheduler::update_draft_token_ids` (request-not-found / finished, and
   `is_prefill_chunk`), so its silence means the ASSIGNMENT
   `request->spec_token_ids = ...` never executes — it does NOT yet distinguish
   "the function is never called" from "it is called and skips every request".
   Ruled out on the way: `post_step` IS on this path (`max_concurrent_batches=1`
   ⇒ `batch_queue_size_ == 1` ⇒ `step_fn_ = &EngineCore::step`, which calls
   `post_step` at `core.cpp:96`), and `check_for_draft_tokens_` is
   `resolved_spec_config_.has_value()`, which the dspark branch of
   `ResolveSpecConfig` satisfies. **Next probe:** a trace at the TOP of
   `update_draft_token_ids` reporting `n`, and per request whether it was
   not-found / finished / `is_prefill_chunk`. That single line decides it.
3. **CORRECTION to the earlier reading.** A first run showed our DSpark-ON text
   byte-identical to our spec-OFF text, and that was reported as the
   self-consistency half of the gate. A second run of the same binary, same
   prompt, temperature 0, produced DIFFERENT text (" Paris.\nA. True\nB. False
   \nAnswer:..." versus " Paris, a city renowned for..."). So the ON path is
   NOT deterministic and the identity was a single-run coincidence, not a
   property. No correctness claim of any kind survives; the earlier one is
   withdrawn. A plausible mechanism to check first: enabling the aux multi-tap
   routes the target through `ForwardDeviceMultiTap`, and that path may not be
   numerically neutral on the 35B MoE NVFP4 the way it is on the gated 27B.
4. **Speed** is 6.78 vs 37.34 tok/s (5.5x slower) — what paying for a drafter
   whose output is discarded looks like.

**ROOT CAUSE FOUND AND FIXED — and it was never DSpark's.**
`EngineCoreProc`'s constructor called the base `EngineCore(scheduler, executor,
structured_output_manager)` WITHOUT the `check_for_draft_tokens` argument, so it
defaulted to `false` on the production path. `post_step` therefore returned at
its first guard, `take_draft_token_ids()` was never pulled, and no speculator's
drafts were ever installed. **This affected EVERY speculator — MTP, DFlash,
ngram, DSpark — through the CLI and the OpenAI server.** The landed spec-decode
gates did not catch it because they drive the engine directly rather than
through `AsyncLLM` -> `InprocClient` -> `EngineCoreProc`.

Fixed by threading the flag `AsyncLLM` -> `InprocClient` -> `EngineCoreProc` ->
`EngineCore` and passing `resolved_spec_config_.has_value()` at the loader's
`AsyncLLM` construction, plus calling `post_step(model_executed)` in the proc
loop's `process_engine_step` where a stale comment had deferred it.

**After the fix, on the 35B gate model, DSpark actually speculates:**

```
[SPECTRACE] req=0 pos=8  k=8 ns=2 acc=1 draft=[3177 421 682 ...] emit=[3177 34756]
[SPECTRACE] req=0 pos=10 k=8 ns=3 acc=2 draft=[364 1141 12761 ...] emit=[364 1141 25438]
```

and throughput goes from **6.78 tok/s (drafts discarded) to 41.89 tok/s**,
against 42.11 spec-OFF — i.e. from 5.5x slower to parity.

**SELF-CONSISTENCY GATE MET (2026-08-10).** The earlier "ON diverges from OFF
and is not run-stable" reading was a CONFOUND, and naming it is the point: the
spec-OFF arm had run with async scheduling ENABLED while the spec-ON arm forces
it off, so the two arms differed in two variables, not one. Re-run with BOTH
arms on the synchronous path (`VT_ASYNC_SCHED=0 VT_ASYNC_RUNNER=0` on the
spec-off arm), 48 greedy tokens, `nvidia/Qwen3.6-35B-A3B-NVFP4` +
`RedHatAI/Qwen3.6-35B-A3B-speculator.dspark` at k=8:

| arm | tok/s | output |
|---|---:|---|
| A spec-OFF, sync | 40.836 | " Paris, a city renowned for its iconic landmarks such as the Eiffel Tower, ... a popular destination for tourists" |
| B spec-OFF, sync (repeat) | 40.916 | **identical to A** |
| C DSpark k=8 | 40.174 | **identical to A** |
| D DSpark k=8 (repeat) | 39.751 | **identical to A** |

So on a gate model, with real acceptance behind it, **speculative-on greedy
output is token-identical to speculative-off and reproducible**. That is the
correctness invariant speculative decoding must satisfy, and it now holds.

**SPEED IS NOT YET A WIN, and is not claimed.** 40.17/39.75 spec-on against
40.84/40.92 spec-off is ~2% SLOWER at c1 on this prompt. The sequential Markov
stage is a host-side loop with a device round-trip per step (spec R5 predicted
exactly this), and acceptance on a bf16-trained draft over an NVFP4 target is
modest. The house gate is at-or-above vLLM's own DSpark-on, so this row stays
`ACTIVE`.

## 6c. MEASURED: what enabling DSpark actually buys (2026-08-10)

The "~2% behind" reading in §6b was a COLD single-shot measurement, where model
load and first-request costs dominate. Warm (`--repeat 3`, c1, 128 tokens, same
target + draft + k, one `flock`), enabling DSpark **does** change the number:

| engine / arm | tok/s | speculative speedup |
|---|---:|---:|
| ours, spec-OFF (warm) | 71.3 | — |
| **ours, DSpark k=8 (warm)** | **~82** (79.5, 84.8) | **1.15x** |
| oracle vLLM 0.25.0, spec-OFF | 25.1 | — |
| **oracle vLLM 0.25.0, DSpark k=8** | **35.4** | **1.41x** |

**Acceptance accounting** (`VT_SPEC_TRACE`, 48 tokens): 18 speculative steps
emitted 48 tokens = **2.67 tokens/step**, of which **1.67 accepted drafts/step**
out of k=8 — a **20.8% acceptance rate**.

**The honest comparison is the RATIO OF RATIOS, not the raw tok/s.** The oracle
arm ran `enforce_eager=True` and one `generate()` per prompt, which is NOT
vLLM's production graphed config (house rule: the denominator is graphed vLLM,
never `--enforce-eager`). Our 82 versus its 35.4 therefore says nothing binding
— our own spec-off 71.3 versus its 25.1 is a 2.8x that the project's own 35B
grid (0.93-1.03x) proves is an artifact of the handicap. What IS like-for-like
is each engine's speculative speedup measured against ITSELF under identical
settings: **upstream gets 1.41x out of this draft, we get 1.15x.**

So DSpark works and pays, but we leave roughly half the available speedup on the
table. The two named suspects, in order: (1) the sequential Markov stage is a
host-side loop with a device round-trip per step, where upstream captures the
WHOLE draft step (backbone + the N-step loop) in one CUDA graph
(`dspark/speculator.py:22-24`); (2) 20.8% acceptance is low for a block drafter
and wants checking against the upstream reference band before it is blamed on
the draft checkpoint's bf16-trained-over-NVFP4 mismatch.

## 6d. DEEP DIVE: the drafter is not the problem (2026-08-10)

Two measurements, both of which contradict the guess in §6c.

**1. Our acceptance is BETTER than upstream's — 2.7x better.** vLLM's own
`spec_decode` metrics on the SAME target + draft + k=8, 64 greedy tokens:

| | drafts | draft tokens | accepted | rate | per-position |
|---|---:|---:|---:|---:|---|
| oracle vLLM 0.25.0 | 39 | 312 | 24 | **7.7%** | `[18, 5, 1, 0, 0, 0, 0, 0]` |
| ours | 18 | 144 | 30 | **20.8%** | 1.67/step |

Upstream accepts past position 2 essentially never. So "20.8% is low, blame the
bf16-trained draft over an NVFP4 target" was wrong: the draft is fine, and our
sampling of it is at least as good.

**2. The draft step is CHEAP, and the sequential Markov loop is 12% of it.**
Phase probe inside the runner's real propose path (`VT_SPEC_TRACE`, 26 steps,
first step discarded as warm-up):

| phase | median | total |
|---|---:|---:|
| parallel backbone block forward | **4.17 ms** | 195.1 ms |
| sequential Markov sampler (k=8) | **0.62 ms** | 26.8 ms |

So the host-side loop with its per-step device round-trip — the thing §6c named
as the prime suspect and the thing upstream's CUDA-graph capture would fix —
costs 0.6 ms of a 4.8 ms draft step. Capturing it could win at most ~1% of the
step. **That lever is dead.**

**Where the speedup actually goes: the VERIFY step.** With the warm numbers
(spec-off 71.3 tok/s = 14.0 ms/step; spec-on ~82 tok/s at 2.67 tokens/step =
32.5 ms/step) and a 4.8 ms draft, the target forward on a speculative step costs
**~27.7 ms against 14.0 ms for a plain one-token decode — about 2x**. A
speculative verify carries 1+k=9 query tokens instead of 1; on a bandwidth-bound
MoE decode that should be nearly free, because it streams the same weights.

If the verify step cost what a plain decode costs, 2.67 tokens per
(14.0 + 4.8) ms would be **142 tok/s, a 2.0x speedup** instead of 1.15x. That
single ratio is the whole gap.

**Next probe (named, not yet run):** whether the T=9 verify falls off the
captured decode CUDA graph into an eager prefill-shaped forward. Our decode
graph is captured for the T=1 decode shape; upstream captures the
speculative-decode shape as a first-class uniform-query case. Confirm with an
`nsys` instance-count check on both arms (the house rule: same tool both sides,
and a captured graph reports as one range), then capture the 1+k shape.

## 6e. nsys: the verify step loses the CUDA graph, AND the MoE is not free (2026-08-10)

Same tool, same binary, both arms, `--cuda-graph-trace=node` (without it a
captured graph collapses to one range and instance counts are meaningless), two
token lengths per arm so decode differences out from the one-time prefill.
Evidence `dgx:~/work/dspark-w6/nsys/`.

**1. The speculative verify runs FULLY EAGER. Proven, not inferred.**

| arm | `cudaGraphLaunch` | `cudaLaunchKernel` |
|---|---:|---:|
| spec-off, 32 tok | **30** | 64,457 |
| spec-off, 64 tok | **62** | 64,617 |
| DSpark, 32 tok | **0** | 70,828 |
| DSpark, 64 tok | **0** | 82,893 |

Graph launches on the spec-off arm scale exactly one per decode step (30 for 32
tokens, 62 for 64). On BOTH DSpark arms there are none at all, and the eager
path adds ~18k extra kernel launches over 64 tokens plus `cuLaunchKernelEx` /
`cuLaunchKernel` families that do not appear spec-off at all. The T=1+k verify
shape is simply not a captured shape.

**2. But the larger factor is MoE expert activation, and this CORRECTS §6d.**
§6d argued the verify "should be nearly free, because it streams the same
weights". That is true of a DENSE decode and FALSE for a top-k MoE. The
dominant decode kernel, per arm (64-minus-32 differencing):

| arm | Marlin MoE instances | avg per instance | per generated token |
|---|---:|---:|---:|
| spec-off | 80 / token | **42.6 us** | 3,408 us |
| DSpark | 37.5 / token | **154.2 us** | 5,783 us |

One token activates top-8 of 256 experts; nine query tokens activate up to nine
different expert sets, so the expert GEMM grows with the query count instead of
riding the same weight stream. Per instance the kernel is **3.6x** more
expensive at T=9, and even after the ~2.1x fewer launches DSpark spends **1.7x
MORE GPU time per generated token** in that one kernel.

**What this means for the row.** The block drafter has to overcome super-linear
expert cost on a sparse MoE, which is a harder bar than on a dense target and
is NOT something better drafting fixes. Two levers, in order:

1. **Capture the 1+k verify shape** (a real, bounded win: the eager path's
   launch overhead is measured above, and upstream treats the speculative-decode
   shape as a first-class captured uniform-query case).
2. **Look at expert batching for the verify shape** — whether the 9 tokens'
   expert sets are being gathered into one grouped GEMM or run as near-separate
   passes. The 3.6x per-instance jump for 9x the tokens suggests the former is
   partly working, but the absolute cost says it is worth measuring against
   vLLM's own MoE path on the SAME shape before assuming.

A dense gate model (Qwen3.6-27B + a dense DSpark draft) would separate the two
factors cleanly, and is the cheapest next experiment.

## 6f. DENSE LANE RESULT: the MoE is the drag, not the graph (2026-08-11)

Qwen3.6-27B NVFP4 (DENSE) + `satgeze/Qwen3.6-27B-DSpark`, k=15, dgx, one flock,
`--max-num-seqs 2` (sized against the #371 state budget: 2.42 GB/slot at k=15,
so the default 32 would have demanded 77 GB).

| arm | warm x3 tok/s | speedup |
|---|---|---:|
| spec-off | 9.846 / 9.878 / 9.886 | - |
| **DSpark k=15** | **17.370 / 17.512 / 17.462** | **1.77x** |

Acceptance: 24 steps, 2.83 tokens/step, 1.83 accepted of k=15 = **12.2%**.
Phase split: backbone 29.8-31.6 ms steady, sequential sampler 9.7-10.3 ms
(~25% of the draft step, against 12% on the 35B).

**THE GRAPH COUNTS OVERTURN 6e's LEADING HYPOTHESIS.**

| arm | `cudaGraphLaunch` |
|---|---:|
| off_32 / off_64 | 30 / 62 |
| **dsp_32 / dsp_64** | **11 / 24** |

The DENSE speculative arm DOES capture CUDA graphs, and the count tracks its
speculative step count (24 launches for 64 tokens at 2.83 tokens/step ~= 23
steps). On the 35B MoE the same arms launched ZERO. So "the speculative verify
falls off the captured decode graph" is NOT a property of the 1+k shape, as
6e proposed - it is specific to the 35B MoE path, and is its own defect.

**Corrected attribution.** With graphs working and no expert term, DSpark returns
**1.77x on a dense target at only 12.2% acceptance**. The 35B's 1.15x was
therefore dominated by MoE EXPERT ACTIVATION (1.7x more GPU time per generated
token in the Marlin MoE kernel at T=9, 6e), not by the missing graph. The lever
worth building is the MoE verify path - expert batching for the 1+k shape, and
finding why that path loses its graph - not a generic 1+k capture.

## 6g. CORRECTION: there is no graph asymmetry (2026-08-11)

Sections 6e and 6f both asserted that the speculative verify "runs fully eager"
on the 35B (zero `cudaGraphLaunch`) while the dense path captured. **That does
not reproduce and the claim is withdrawn.** Re-measured on the current binary
with the dense lane's recipe (`--max-num-seqs 2`, same tool, same two lengths):

| arm | `cudaGraphLaunch` |
|---|---:|
| 35B spec-off 32 / 64 | 30 / 62 |
| **35B DSpark 32 / 64** | **15 / 28** |
| 27B spec-off 32 / 64 | 30 / 62 |
| **27B DSpark 32 / 64** | **11 / 24** |

`VT_DFLASH_GRAPH_STATS=1` confirms it independently: `[DFLASH-GRAPH] captured
#1 Tq=8 C=5` (35B) and `Tq=15` (27B). BOTH families capture. Issue #389, filed
on the non-reproducing zero, is CLOSED as not a defect.

**What the gate actually says**, which reading it would have shown before either
section was written: both target gates require `input.pure_decode`
(`num_actual_tokens == num_reqs`; qwen3_5_dense.cpp:159, qwen3_5_moe.cpp:128).
A verify carries 1+k tokens per request, so **the verify is eager on BOTH
families** — including the dense one that reached 1.77x. The graph launches under
DSpark are the DRAFT step (D13 Part C, gated `P == 1`), one per propose, which is
exactly why the count tracked speculative steps.

**Net effect on the attribution: it survives and gets cleaner.** With no graph
difference between the families, the 35B's 1.15x against the dense 1.77x rests
entirely on MoE expert activation (1.7x more GPU time per generated token at
T=9, 6e). And the standing lever is better evidenced than when 6e proposed it:
the 1+k verify is eager EVERYWHERE, so capturing that shape is unexploited
headroom on both families rather than a repair to one.

**Method note worth keeping:** 6e drew an architectural conclusion from a single
profile without a same-binary control. The dense lane was the control that caught
it, and the counter settled it in one run. A launch-count difference between two
models is a claim about a gate — read the gate.

**Still owed for a binding W6:** (1) the cross-engine speed A/B against vLLM's
PRODUCTION GRAPHED config through the project's harness, not the eager arm run
here; (2) token-exactness against that oracle on the SACRED corpus rather than
ad-hoc prompts; (3) the acceptance-rate band against the upstream reference;
(4) the 27B lane and the Gemma4 `1 + N` layout; (5) capturing the SPECULATIVE VERIFY shape, which § 6d measures as the real
gap (the draft-step capture that looked like the lever is worth ~1%).

## 7. Evidence, authority, stop conditions

- Evidence root: `dgx:~/work/vllm.cpp-dspark-<slice>/`, one `flock`, named tmux.
- Authority still needed from the developer: (a) **downloading the draft +
  target checkpoints** (2.8–8.8 GB each — a large-asset download), (b) GPU-box
  time on dgx, (c) pushing `row/SPEC-DSPARK` and opening its draft PR. Until (a)
  lands, W1/W2/W4 remain CPU-gateable on synthetic fixtures and R1 is `PENDING`.
- Stop conditions: R1 unprovable at the pin (oracle cannot run DSpark) → record
  the exact external blocker and stop; any slice failing its red-then-green
  discipline → back to a fresh implementer, never repaired in the coordinating
  session (POL-REVIEW-NO-REPAIR).

Sources: pin files cited inline at `555967922`; HF API queried 2026-08-09;
our anchors cited against `bc6e3d72`.
