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
PRODUCTION GRAPHED config — **DONE in §6h**, against the pinned oracle;
(2) token-exactness against that oracle on the SACRED corpus rather than ad-hoc
prompts; (3) the acceptance-rate band against the upstream reference — **DONE
in §6h**, and it is the finding: 35B matches (20.8% vs 20.4%), the 27B dense
lane does not (12.2% vs 49.3%); (4) the 27B lane — **measured in §6h** — and
the Gemma4 `1 + N` layout, still unrun on real weights; (5) capturing the
SPECULATIVE VERIFY shape, which § 6d measures as the real gap (the draft-step
capture that looked like the lever is worth ~1%).

## 6h. CROSS-ENGINE A/B vs the PINNED, GRAPHED oracle (2026-08-11)

Every earlier oracle arm in this spec ran `enforce_eager=True`, which the house
rule forbids as a denominator. This is the binding replacement: vLLM in its
production graphed configuration, same target + draft + k + `max_num_seqs=2`,
same greedy sampling, one `flock`, both prompts timed per-prompt.

**The first attempt measured the WRONG ORACLE.** `~/venvs/vllm-oracle` is a
symlink to the preserved `v0.25.0-stage` ROLLBACK, not to the pin, so the run
recorded `vllm 0.25.0` and proceeded happily — issue #375's failure mode
exactly, and a deterministic wrong reference is indistinguishable from a right
one. The pinned build is `~/venvs/vllm-oracle-next`
(`0.23.1rc1.dev1511+g555967922`, FlashInfer 0.6.15.post1, Torch 2.13.0,
transformers 5.14.1 — matching `upstream-sync.md` on every axis). The harness
now ASSERTS commit + FlashInfer and aborts on mismatch. The 0.25.0 numbers are
kept only as context; they are NOT quotable, and they mislead in both
directions (the rollback's DSpark is far slower: 35B 89.8 vs the pin's 146.4).

Warm tok/s (`completion_tokens / whole-request wall`, identical on both sides:
`examples/cli/main.cpp:253-270` vs the oracle's `generate()` wall), median of
warm repeats, cold first run discarded:

| lane / prompt | ours | pinned vLLM | ours/vLLM | our speedup | its speedup |
|---|---|---|---|---|---|
| 35B spec-off, "capital" (128 tok both) | 71.47 | 73.91 | 0.967x | — | — |
| 35B DSpark k=8, "capital" (128 tok both) | 74.10 | 75.92 | 0.976x | 1.037x | 1.027x |
| 35B DSpark k=8, "fibonacci" (89 tok both) | 134.85 | 146.41 | 0.921x | 1.914x | 2.583x |
| 27B spec-off, "fibonacci" (126 tok both) | 9.80 | 9.51 | 1.031x | — | — |
| 27B DSpark k=15, "fibonacci" | 31.83 | 34.06 | 0.935x | 3.248x | 3.583x |
| **27B DSpark k=15, "capital"** | 17.41 | **49.71** | **0.350x** | 1.757x | **5.233x** |

Acceptance, from upstream's own `vllm:spec_decode_*` counters against ours:

| lane | ours | pinned vLLM | accepted per draft (vLLM) |
|---|---|---|---|
| 35B A3B MoE, k=8 | 20.8% | 20.4% (212/1040) | 1.63 of 8 |
| 27B dense, k=15 | 12.2% | **49.3%** (281/570) | **7.39 of 15** |

**Result.** Two different verdicts, and the previous framing had them backwards.

- **35B MoE lane: near parity.** 0.92–0.98x cross-engine, and our acceptance
  MATCHES upstream's (20.8% vs 20.4%). Nothing here points at draft quality;
  the residual is per-step cost, the same story §6f measured.
- **27B dense lane: the real gap, and it is DRAFT QUALITY.** Upstream accepts
  49.3% where we accept 12.2% — 7.39 accepted tokens per draft against our
  ~1.8 — so upstream extracts 5.23x from the same checkpoint where we get
  1.76x. §6f called this lane's 1.77x our strongest result; measured against
  the actual reference it is our weakest. That 1.77x was a self-speedup, never
  a cross-engine one.

**Ruled out for the dense acceptance gap** (checked, not assumed): mis-resolved
aux taps. The 27B draft declares five taps at absolute target indices
`[1, 16, 31, 46, 61]` and the 27B target is a hybrid (linear-attention +
full-attention layers), so a tap resolved against a filtered layer list would
land on the wrong layer and degrade drafts silently. Both `MaybeCaptureAuxTap`
call sites sit inside the full `for (l = 0; l < num_hidden_layers; ++l)` loop
and index by absolute `l`, covering both layer kinds
(`qwen3_5.cpp:6707`, `:7546`). Also ruled out: the Markov step loop, which
matches upstream's shape (`prev` seeded from the anchor, re-biased per step,
argmax over `base + bias`, then `d2t` — `speculator.py:120-121,148`).

**Next traceable hypothesis** (no ceiling is being declared): the 27B draft uses
the flat/native config layout and carries NO `draft_vocab_size`, where the 35B
carries 32000 and reaches upstream-matching acceptance through the Speculators
translation path. The d2t/reduced-vocab handling on the flat path is therefore
the next thing to instrument — compare our proposed draft token ids against
upstream's for an identical prefix, rather than comparing acceptance totals.

**Caveat on the table.** Greedy outputs diverge between arms and between
engines on several cells, so completion counts differ (27B "fibonacci": ours
128, upstream 52; 27B "capital": ours 84, upstream 128). Per-token throughput
still compares, but the content differs. The cleanest cells — where all arms
returned identical token counts — are 35B "capital" (128 everywhere) and 35B
DSpark "fibonacci" (89 both), and those are the 0.92–0.98x readings. The 27B
"capital" 0.350x is a 2.9x gap that content differences cannot account for.

Note for the token gate: on the pinned oracle the 35B spec-on and spec-off
streams are byte-identical (`[11751, 11, 264, 3177, 34756, ...]` both), i.e.
upstream's DSpark is lossless on that lane, while its 27B arms diverge at
position 3. Ours returned different completion counts between arms on 35B
"fibonacci" (87 off vs 89 on) where upstream returned 89 for both.

Tracked as issue #430. Evidence: `dgx:~/work/dspark-w6/pinned_{35b,27b}_{on,off}.json` (pinned),
`xengine_*.json` (0.25.0, non-binding), `xengine.log`, `xengine_ours.log`.

## 6i. TEXT PARITY vs the pinned oracle: the BASELINE drifts too (2026-08-11)

§6h flagged that our 35B "fibonacci" arms returned 87 tokens spec-off and 89
spec-on where upstream returned 89 for both, and asked whether our DSpark is
lossy. Measured, on the same eight arms, greedy, against the pinned oracle's
own text:

| case | verdict | first divergence |
|---|---|---|
| 35B spec-OFF "capital" | DIFFER | char 218 ("the 10th century" vs "the Middle Ages") |
| 35B spec-OFF "fibonacci" | DIFFER | char 91 |
| 35B DSpark "capital" | DIFFER | char 76 |
| 35B DSpark "fibonacci" | DIFFER | char 55 (`return (` vs `return(`) |
| 27B spec-OFF "capital" | DIFFER | char 244 |
| 27B spec-OFF "fibonacci" | **EXACT** | 382 chars |
| 27B DSpark "capital" | DIFFER | char 7 |
| 27B DSpark "fibonacci" | DIFFER | char 5 |

**The answer is no, and the question was malformed.** Our SPEC-OFF baseline
already diverges from the pinned oracle's spec-off on three of four prompts, so
an ad-hoc prompt comparison cannot isolate a DSpark defect: there is nothing to
subtract it from. The divergences are the ratified near-tie regime (identical
for 218 and 244 characters, then a single word choice diverging and the
trajectories separating), which is exactly what the distributional gate exists
for. Nothing here is attributable to the speculator.

Two consequences:

1. The completion-count asymmetry noted in §6h is withdrawn as evidence about
   DSpark. It is baseline drift.
2. The owed token gate is unchanged and cannot be replaced by this check: it
   must be the SACRED corpus under the ratified near-tie/distributional
   protocol (ours in the oracle's K-run set), not ad-hoc prompts. What §6i adds
   is the reason a cheap substitute will not do.

Note the two 27B DSpark arms diverge very early (chars 7 and 5) while the 27B
spec-off arm on the same prompt matched for 382 characters. That is consistent
with the #430 acceptance gap rather than independent of it, but it is NOT
evidence on its own, because the other 27B prompt's spec-off arm diverged at
char 244. Instrumenting proposed draft ids against upstream's (the #430 next
step) settles it; this does not.

Evidence: `dgx:~/work/dspark-w6/parity/*.txt` vs `pinned_*.json`.

## 6j. CORRECTION: there is no draft-quality gap; the gap is PER-STEP COST (2026-08-12)

Section 6h concluded "27B dense lane: the real gap, and it is DRAFT QUALITY",
from upstream's aggregate counters (49.3% accepted) against ours (12.2%). Direct
instrumentation of BOTH engines refutes that. Issue #430 is closed as refuted.

**Our proposals are nearly token-identical to upstream's.** Dumping upstream's
`_sample_sequential` (anchor + the k proposed ids) against our `VT_SPEC_TRACE`
at the same anchors:

| anchor | upstream proposes | we propose |
|---|---|---|
| 11751 | `13 271 9764 6511 220 6511 314 279 369 279 198 13 198 ...` | `13 271 9764 6511 220 6511 314 279 369 279 198 13 198 ...` |
| 248068 | `198 248069 271 760 6511 314 9338 369 ...` | `198 248069 271 760 6511 314 9338 369 ...` |
| 2972 | `57590 332 369 279 6511 314 9338 9338 13 248046 ...` | `57590 332 369 279 6511 314 9338 9338 13 248046 ...` |

**Acceptance is at parity on matched content.** Same prompt, both engines
emitting the same tokens: ours 32 tokens in 11 draft steps (12.1% of k=15),
upstream 32 tokens in 12 steps (11.1%). We are marginally ahead.

**Where 49.3% came from.** Upstream's graphed benchmark run on "The capital of
France is" DEGENERATED into a repetition loop: **8 distinct token ids across 128
tokens** ("The capital of France is Paris." ten times), while upstream's OWN
spec-off arm on that prompt produced ordinary reasoning text. Trivially
predictable text is trivially draftable, which is what produced 7.39 accepted
per draft and 49.7 tok/s. Our engine generated real reasoning text there, so the
"0.350x" cell compared 84 tokens of reasoning against 128 tokens of a loop. It
was never a like-for-like cell, and §6h's caveat understated that badly.

**Ruled out for the acceptance question** (each checked in source, not assumed):
aux tap indexing; the Markov step loop; d2t / reduced vocab (this checkpoint has
NO d2t — both Markov tables are full `[248320, 256]`); `logit_scale` (upstream
routes base logits AND bias through the same `logits_processor`, so it cancels
in the argmax); head aliasing (we correctly keep the draft's own shipped
`embed_tokens`/`lm_head`); **risk R3 RESOLVED** — `attn_output_gate` is inert
metadata, `q_proj` is `[4096, 5120]`, not doubled; and the non-causal resolution
(ours mirrors upstream's inverted `causal = is_sliding` rule, so all five draft
layers are non-causal exactly as upstream has them).

**The anchor layout is CORRECT.** An A/B on our own engine, same weights with
only `sample_from_anchor` flipped in a copied config, drives acceptance to ZERO
on every step when forced to the 1+N layout (15.6 -> 6.24 tok/s): the proposals
land off-by-one because that layout drops the anchor row's prediction. The 27B
is the only lane on `sample_from_anchor=true` (the flat config defaults it true,
`model_loader.cpp:407-408`; the Speculators translation writes false explicitly,
`qwen3_dspark.cpp:133-136`), which is why it was the natural suspect. It is not
the defect.

**What the gap actually is.** `[spec-phase] backbone=27.44ms sample=10.50ms`:
28% of the draft step is host-side sampling, because every block-forward variant
returns `std::vector<float>` and the host loop then downloads a
`[B, draft_vocab]` bias and argmaxes over the FULL vocab once per step. On the
27B that is k x 248320 floats per step; the 35B's reduced 32000-vocab draft
moves 13x less, which is part of why THAT lane already sits at 0.92-0.98x. This
is spec risk R5, recorded when the lane landed and never closed. W7 (#436)
closes it.

## 6k. W7 RESULT: device sampling is token-identical, and the residual is a BANDWIDTH bound (2026-08-12)

A/B on the same binary, only `VT_DSPARK_DEVICE_SAMPLE` differing, one flock,
warm repeats, cold run discarded.

| arm | sample ms | warm tok/s |
|---|---|---|
| 27B host loop | 10.44 | 17.31 |
| 27B device | **9.24** (-11.5%) | 17.42 |
| 35B host loop | 0.59 | 71.06 |
| 35B device | **0.50** (-15%) | 73.3 (+3.2%) |

**Text is byte-IDENTICAL on both lanes**, which is the bar for a pure cost move.

**The residual is not host overhead, it is bandwidth, and upstream pays it too.**
`markov_w2` is `[draft_vocab, markov_rank]` bf16, so the per-step bias GEMV
re-reads the WHOLE matrix:

| lane | markov_w2 | per draft step | at ~205 GB/s | measured |
|---|---|---|---|---|
| 27B (V=248320) | 127 MB | 15 x 127 MB = 1.9 GB | 9.3 ms | **9.24 ms** |
| 35B (V=32000) | 16 MB | 8 x 16 MB = 131 MB | 0.6 ms | **0.50 ms** |

Two lanes 13x apart in vocab both landing on the bound is not a coincidence. The
k sequential GEMVs are required by DSpark's mechanism (`speculator.py:151`),
upstream runs the same math, and no amount of graph capture removes weight
traffic. **Do not chase this further as if it were overhead**: the only way to
move it is to change what is computed (pruning the bias to a candidate set),
which changes tokens and is out of scope for a cost move.

What W7 removed is the host part: k downloads of `[num_reqs, draft_vocab]` f32
and k host argmaxes over the full vocab, replaced by one base-logits upload and
a `[num_reqs]` i64 readback per step. Worth 11-15% of the sampling phase and
~3% e2e on the 35B lane.

The block-forward's own `[nqpr, draft_vocab]` D->H, which every variant still
performs (`qwen3_dflash.h:155,202,248,284`), sits inside `backbone=27.4ms` and is
NOT addressed here. At 14.9 MB it is worth ~0.15 ms on GB10, so it is a small
prize and should be SIZED before it is built.

Evidence: `dgx:~/work/dspark-w6/w7_ab.log`, `w7/*.txt`.

## 6l. PAIRED PARITY RE-MEASURE WITH W7: still BELOW parity (2026-08-12)

Both engines on an idle box after a reboot, pinned graphed oracle, same
target+draft+k, `max_num_seqs=2`, greedy, matched token counts, cold run
discarded. Ours is 6 reps (median of the 5 warm).

| 35B cell | ours | pinned graphed vLLM | ratio |
|---|---|---|---|
| "capital", 128 tok both | 75.82 | 77.28 | **0.981x** |
| "fibonacci", 89 tok both | 135.39 | 155.60 | **0.870x** |

Our fibonacci reps are 134.81 / 135.41 / 135.24 / 135.39 / 135.40 — a 0.4%
spread — so 0.870x is a real reading, not noise. The capital cell spreads ~8%
(71.58-77.67), so treat 0.981x as the looser of the two.

**W7 did not close the gap, and the goal of speed parity is NOT met.** W7 removed
host-side sampling waste worth 11-15% of the sampling phase, but sampling is a
small share of the step on this lane (0.5 ms of it), so the e2e effect is within
the capital cell's own noise band. Do not read the earlier 79.19 tok/s as a W7
win: that was PRE-REBOOT machine state, and on the idle box both engines moved,
the oracle more than us. Same-session pairing is what makes these two rows
quotable and the earlier cross-session ones not.

**Where the remaining gap is, and the next lever.** The deficit is largest in the
HIGH-ACCEPTANCE regime (fibonacci 0.870x, where the block is mostly accepted and
the T=1+k verify runs every step) and smallest in the low-acceptance one
(capital 0.981x). That points at the verify forward, not the drafter: §6g
established that BOTH model families gate their decode CUDA graph on
`input.pure_decode` (`qwen3_5_dense.cpp:159`, `qwen3_5_moe.cpp:128`), so our
speculative verify at T=1+k falls off the captured graph and runs EAGER, while
upstream captures the uniform `1+k` shape. That is the same unexploited headroom
DFlash left open (D12 Part C) and it is the next thing to build. It is a
static-shape capture increment, not a tuning knob.

Evidence: `dgx:~/work/dspark-w6/parity35b.log`, `pinned_35b_on.json`.

## 6m. W8 SLICE: capture the T=1+k verify (the remaining gap), issue #442

§6l localises the residual 0.870x-0.981x to the VERIFY forward. This is the
scoped increment that closes it; it is dispatch-ready and deliberately NOT
started here, because a half-finished CUDA-graph capture is worse than none.

**ROOT CAUSE (2026-08-12): we mirrored the model but NOT vLLM's cudagraph
dispatcher.** Upstream's "uniform" test is that all requests share a query_len,
NOT that query_len is 1 (`v1/worker/gpu/cudagraph_utils.py:95-105`), and the
captured decode length is defined as the verify shape:
`self.uniform_decode_query_len = 1 + self.vllm_config.num_speculative_tokens`
(`v1/cudagraph_dispatcher.py:37`), so a uniform 1+k batch takes the FULL graph
(`:143-146`). vLLM graphs the speculative verify BY CONSTRUCTION. We never do,
so our verify runs eager EVERY step — the whole ~4.8 ms/step, and the reason the
deficit tracks acceptance (0.870x where the block is mostly accepted, 0.981x
where it is not). Nothing is missing from the model: kernels, drafter and
acceptance are already at parity. The missing piece is dispatch.

This makes the increment a MIRRORING job with an upstream anchor: generalise the
predicate to upstream's uniform test with query length `1 + num_spec`, key
captures on a `(num_tokens, num_reqs, uniform)` descriptor instead of one bespoke
pure-decode graph, and pad token counts to a captured set as
`_bs_to_padded_graph_size` does so a handful of shapes covers every batch.

**Predicate.** `pure_decode` is `num_actual_tokens == num_reqs`
(`qwen3_5_dense.cpp:159`, `qwen3_5_moe.cpp:128`). A verify submits
`num_reqs x (1+k)` tokens and therefore runs eager EVERY step, while upstream
captures the uniform `1+k` shape.

**Do NOT just widen the gate.** `Qwen3_5DenseDecodeGraph` is documented
pure-decode ("all query_len==1"); relaxing the predicate would send a spec batch
through a graph captured for a different shape AND a different attention path.
The increment is a sibling capture keyed on `(num_reqs, 1+k)` that routes the
SPEC paths already landed under `SPEC-GDN-SEGMENTS` (`vt::GdnSpecDecode`,
`vt::CausalConv1dSpecUpdate`, per-timestep snapshots) and block-diagonal causal
attention over the query span. k is fixed per config and `num_reqs <=
max_num_seqs`, so the shape set is small and static.

**Gate:** replayed == eager BIT-IDENTICAL on both gate models; spec-OFF
byte-identical (SACRED); then the paired cross-engine re-measure on the same two
35B cells, target >= 1.0x on both.

**Capture safety:** no function-local upload temporaries in the captured region.
This repo has already shipped a use-after-free from exactly that, and a
sanitizer-clean run is not proof of capture safety -- pair it with an explicit
replay-vs-eager bit-compare.

## 6n. W8 LANDED: the verify is captured, and the 35B lane reaches ~parity (2026-08-12)

§6l put the MoE lane at 0.870x / 0.981x and localised the deficit to the eager
T=1+k verify. §6m named vLLM's dispatch predicate as the missing mirror. This is
the implementation and its measurement.

**Paired against the pinned graphed oracle, one lock session, matched token
counts, ours = median of 3 warm reps:**

| 35B cell | before W8 | with capture | pinned vLLM | ratio |
|---|---|---|---|---|
| "capital", 128 tok both | 72.23 | **78.37** | 78.76 | **0.995x** (was 0.981x) |
| "fibonacci", 89 tok both | 134.55 | **140.82** | 142.88 | **0.986x** (was 0.870x) |

Capture is worth **+8.5%** and **+4.7%** on the SAME binary
(`VT_SPEC_DECODE_GRAPH=0` is the eager arm), and it closes the high-acceptance
cell from 0.870x to 0.986x, which is where the ~4.8 ms/step lived. Not >= 1.0x:
the residual is ~1.4% and our reps spread 0.3%, so it is real, not noise.

**Correctness.** Text byte-identical capture-vs-eager on both lanes, and the
project's four e2e spec-decode suites pass with capture ON and OFF with identical
assertion counts: `test_qwen27_spec_decode` 9, `test_qwen27_dflash_spec_decode`
27, `test_qwen36_spec_decode` 9, `test_qwen27_spec_decode_concurrent` 5. Default
ON re-verified with the env unset.

**What it took, because four of the five attempts were wrong and each failure
taught the next.**

| attempt | result | cause |
|---|---|---|
| predicate only | INERT, no graph built | `ForwardDeviceMultiTap` returns BEFORE the gate |
| + aux capture | both arms crashed | padding rewrote spec metadata; Step's fallback dropped aux |
| + those fixes | captured, but GARBAGE + 5x SLOWER | stale `num_accepted` |
| + spec staging | `cudaMemcpyAsync: invalid argument` | per-request arrays sized by TOKEN count |
| + token/request split | correct AND faster | — |

Three of these deserve to be remembered:

1. **The aux tap is why the verify never captured.** DFlash/DSpark must emit the
   `[T, H*taps]` hidden capture the drafter conditions on, and that path returned
   before the decode-graph gate. No predicate change could have reached it. Each
   `SizeSlot` now owns a persistent aux buffer, like its logits.
2. **Stale `num_accepted` is silent and catastrophic.** `StageStepInputs` refilled
   only the pure-decode fields, so a replay re-read the PREVIOUS step's
   acceptance — the value `GdnSpecDecode` uses to pick the initial state (column
   `num_accepted-1`) and advance the conv window. Wrong state produced incoherent
   tokens AND 5x slower (26 vs 136 tok/s), because zero acceptance multiplies the
   step count. `StageSpecStepInputs` refills them per step, outside the region.
3. **We had collapsed TOKENS and REQUESTS into one count.** The staging sized
   `block_table` as `S*cols`, `seq_lens` as `S`, `qsl` as `S+1` — correct only
   because pure decode has `S == num_reqs`. Under speculation `S = num_reqs*(1+k)`,
   so it overran host and device buffers. Upstream carries BOTH in its graph key
   (`BatchDescriptor(num_tokens, num_reqs, uniform)`) for exactly this reason.
   This is the deepest form the "mirror vLLM" rule took here: the divergence was
   not a kernel or a heuristic, it was a data model.

A false pass to remember too: an early run reported text `IDENTICAL` while
comparing two EMPTY outputs, because both arms had failed identically. The exit
codes caught it; the diff did not.

**Owed next:** the residual ~1.4%, the 27B dense lane re-measure (its cells were
never like-for-like), and the padded/multi-request spec shapes — capture takes the
EXACT shape today, which is bounded by `max_num_seqs` but leaves batching on the
table.

## 6o. METHOD CORRECTION: the oracle is not a stable denominator single-shot (2026-08-12)

§6n quoted 0.986x / 0.995x from ONE oracle run per cell. That method does not
survive contact with this box, and the corrected numbers are worse.

**The oracle's own result moves up to 27% between same-config sessions:**

| cell | oracle 12:23 | oracle 14:58 |
|---|---|---|
| "capital" | 78.76 | 98.01 |
| "fibonacci" | 142.88 | 152.45 |

Our reps hold to 0.3% across the same span, so the variance is the reference, not
us. Part of it was self-inflicted: the 14:58 run put the oracle FIRST to dodge a
GB10 reboot during its load, which handed it a freshly booted idle box — its best
slot. The same comparison therefore reads 0.995x/0.986x one way and
0.833x/0.925x the other.

**INTERLEAVED (O,U,O,U,O,U in one flock, medians) is the binding form:**

| cell | ours (per-rep medians) | oracle (per rep) | ratio |
|---|---|---|---|
| "capital", 128 tok | 75.3, 81.4, 78.2 | 81.5, 85.1, 97.8 | **0.919x** |
| "fibonacci", 89 tok | 141.4, 141.4, 141.4 | 143.2, 149.4, 142.1 | **0.987x** |

**So the lane is 0.92x-0.99x, NOT parity.** §6n's 0.986x/0.995x is superseded and
should not be quoted. What survives §6n unchanged is the same-session,
same-binary A/B, where both arms run adjacent under identical conditions and the
capture is the only difference:

| cell | capture OFF | capture ON | delta |
|---|---|---|---|
| "capital" | 72.7 | 81.6 | **+12.2%** |
| "fibonacci" | 136.2 | 141.0 | **+3.5%** |

That, the byte-identical output and the four green e2e suites are what justify
W8; the cross-engine ratio is a separate claim and it is not yet met.

**Rule for this row from here:** no cross-engine ratio without interleaved
repetitions and medians. A single oracle load is worth nothing on a box whose
reference swings 27%, and the direction of the error depends on who got the first
slot.

## 6p. FINAL MEASUREMENT: the oracle's own acceptance is non-deterministic (2026-08-12)

5-rep INTERLEAVED (O,U x5 in one flock), medians of per-rep medians:

| cell | ours (median, range) | oracle (median, range) | ratio |
|---|---|---|---|
| "capital", 128 tok | **78.04** [76.0-79.3] | 77.16 [74.6-96.5] | **1.012x** |
| "fibonacci", 89 tok | **141.83** [138.9-142.4] | 149.03 [142.1-151.6] | **0.952x** |

The 3-rep run gave 0.919x / 0.987x for the same cells. Both are "correct"; the
ratio is simply not stable, and this is why:

| oracle rep | fib tok/s | capital tok/s | drafts | acceptance |
|---|---|---|---|---|
| 1 | 149.03 | 77.16 | 127 | 21.2% |
| 2 | 151.61 | 74.60 | 117 | 24.0% |
| 3 | 142.08 | 76.84 | 124 | 22.6% |
| 4 | 151.13 | 78.50 | 127 | 21.2% |
| 5 | 142.31 | **96.48** | **104** | **29.6%** |

**Upstream's speculative decode is NOT run-to-run deterministic.** Same prompts,
same greedy sampling, identical output lengths every rep (89 / 128 tokens), yet
its draft count moves 104-127 and its acceptance 21.2-29.6%. Fewer draft steps is
directly fewer forwards, which is the 96.48 outlier and the 142-vs-151 bimodality
on fibonacci. OURS is deterministic: identical tokens in an identical number of
steps every run, which is why our range is 76-79 and 139-142.

So a point ratio against this reference measures WHICH DRAW WE CAUGHT as much as
either engine. Read distributionally instead:

- "capital": ours 78.04 sits ABOVE the oracle's median (77.16) and above 3 of its
  5 draws.
- "fibonacci": ours 141.83 sits just BELOW the oracle's floor (142.08), i.e.
  0.998x of its worst draw and 0.952x of its median.

**Verdict: approximate parity, cell-dependent, and NOT a clean >= 1.0x on both
cells.** The row does not claim parity. This is the same condition the project
already ratified for token-exactness — where the oracle's own greedy is
non-deterministic the gate is distributional — now showing up on the SPEED axis,
and the honest form of the speed claim is a distribution, not a single ratio.

**Per-step, the engines are aligned** (§ diag): ours 30.4 ms/step on "capital"
vs the oracle's ~30.1, and 34.7 vs ~34.5 on "fibonacci", with our draft graph
capturing (`[DFLASH-GRAPH] replays=96 captures=2`), the verify captured, and the
Markov sample at its bandwidth bound. There is no structural gap left to close;
what remains is inside the reference's own spread.

## 6q. THE FIBONACCI GAP RESOLVED: identical acceptance, one lucky token (2026-08-12)

§6p reported 0.952x on "fibonacci" against the oracle's 5-rep median. Isolating
that prompt (only prompt, warm-up on the same prompt, so the cumulative
`spec_decode` counters are attributable) explains it and removes it.

Upstream, per run:

| run | tok/s | drafts | accepted | rate | tokens/step |
|---|---|---|---|---|---|
| 0 | 142.01 | 18 | 70 | 48.6% | 4.94 |
| 1 | 142.66 | 18 | 70 | 48.6% | 4.94 |
| 2 | 142.37 | 18 | 70 | 48.6% | 4.94 |
| 3 | **151.18** | **17** | **71** | 52.2% | 5.24 |

The 151 draw is ONE EXTRA ACCEPTED TOKEN (71 vs 70), which removes one whole
draft step (17 instead of 18). It happened in 1 run of 4. Nothing about that draw
is faster per step; it simply did less work.

**Our acceptance is IDENTICAL to upstream's modal value**: 48.6%, 4.94
tokens/step, 18 steps (§ diag: `accepted/step mean=3.89 of k=8`, 36 SPECTRACE
lines over 2 reps). So on matched work:

| comparison | ours | upstream | ratio |
|---|---|---|---|
| vs its MODAL draw (3 of 4 runs) | 141.83 | 142.01-142.66 | **0.996x-0.999x** |
| vs its 5-rep median (inflated by 2 lucky draws) | 141.83 | 149.03 | 0.952x |

The 0.952x in §6p is therefore an artifact of averaging over a bimodal reference,
not a deficit. **Read against matched work the MoE lane is 0.996x-1.012x**:
"capital" 1.012x (§6p, ours above the oracle's median and 3 of its 5 draws) and
"fibonacci" 0.996x-0.999x here.

That is parity within the measurement's own resolution. It is NOT a demonstrated
>= 1.0x on both cells -- "fibonacci" sits 0.1-0.4% under upstream's modal draw,
which is smaller than either engine's run-to-run spread -- so the row records
parity-within-noise rather than a win, and the remaining honest statement is that
neither engine is reliably faster on this workload.

**Method note worth keeping:** a bimodal reference must be compared MODALLY or
per-work, not by its mean/median. Three of this row's ratios (0.986x single-shot,
0.919x/0.987x at 3 reps, 0.952x at 5 reps) were all measuring the reference's
draw distribution rather than either engine, and only isolating the prompt and
reading its counters settled it.

## 6r. STOP CONDITION: the residual is below the measurement's resolution (2026-08-12)

The last cell under 1.0x is "fibonacci" at 0.996x of upstream's modal draw. That
difference is not resolvable on this box, and no remaining lever can close it.

| quantity | value |
|---|---|
| ours, 5 reps | min 138.90, median 141.80, max 142.40 |
| oracle, modal draws (18 steps / 70 accepted) | min 142.01, median 142.37, max 142.66 |
| **distributions** | **OVERLAP** — our max 142.40 exceeds their min 142.01 |
| median ratio | 0.9960x |
| best-vs-best | 0.9982x |
| OUR run-to-run spread | **2.47%** |
| their modal spread | 0.46% |

The 0.4% median difference is SIX TIMES SMALLER than our own spread, and the two
distributions overlap, so the ordering is not established in either direction.

**And no identified change can close it.** The largest remaining lever is
dropping the block-logits round trip (the forward downloads
`[nqpr, draft_vocab]` = 1.15 MB and the sampler uploads it straight back). The
sync it implies has to happen anyway, so the saving is ~0.03 ms of a 34.7 ms
step = **0.09%**, which would move 0.996x to ~0.997x. Everything else is already
at its floor: the draft graph captures, the verify captures, the Markov sample is
at its memory-bandwidth bound (§6k), and per-step the engines match (30.4 vs
~30.1 ms, 34.7 vs ~34.5).

**Verdict for this row: parity within the measurement's resolution.** "capital"
1.012x, "fibonacci" 0.996x with overlapping distributions, acceptance identical
at 48.6% / 4.94 tokens per step. Neither engine is reliably faster on this
workload, and the row claims neither.

**What a demonstrated >= 1.0x would now require** is a lower-noise harness, not
more engineering: pinned clocks, many more repetitions, and a reference whose own
acceptance does not vary run to run (upstream's moves 104-127 drafts on identical
greedy prompts, §6p). That is a benchmarking task, and it is the honest next step
for anyone who needs the strict claim.

## 6s. LOW-NOISE HARNESS: the gap is REAL, 0.975x, and §6q/§6r were too generous (2026-08-12)

§6r asserted the residual was below the measurement's resolution and stopped. A
controlled harness disproves that. Clocks PINNED at 1800 MHz, 16 reps per arm
with the cold run dropped, our arm run TWICE to bracket the oracle so drift is
detectable, and the oracle's non-modal draws excluded by their own draft counts:

| arm | n | median | range | spread |
|---|---|---|---|---|
| ours, BEFORE | 15 | 135.98 | 135.3-136.2 | **0.66%** |
| ours, AFTER | 15 | 135.86 | 128.8-136.0 | 5.30% |
| drift before -> after | | **-0.088%** | | |
| oracle, MODAL (18 steps) | 12 | **139.36** | 137.8-139.7 | 1.33% |
| oracle, non-modal (17 steps) | 3 | 147.7-148.2 | | excluded |

**The distributions do NOT overlap** — our max 136.2 is below their min 137.8 —
drift is negligible (-0.088%), and the ratio is **0.9751x** by median and 0.9750x
best-vs-best.

So the honest number for this cell is **0.975x, not 0.996x**, and §6q/§6r are
superseded. What went wrong there was not arithmetic but CONDITIONS: at free
boost clocks the two engines read 141.8 vs 142.4 (0.996x), and pinning the clock
moved ours to 135.9 and theirs to 139.4 (0.975x). Both slowed, ours MORE.

**That clock sensitivity is itself the lead.** Lowering the SM clock hurt us
disproportionately, which means our step carries more SM-clock-sensitive
(compute-bound) work per token than upstream's — a memory-bound step would have
been comparatively unaffected. That points back at a finding this row already
made and never closed: §6e/§6f measured the MoE expert activation at T=9 costing
us ~1.7x GPU per token. The verify is ~30 ms of a ~34.7 ms step and both engines
graph it, so the residual is inside the MoE expert path, not in launch overhead.

**Method lesson, and it is the inverse of the last one.** §6p/§6q corrected an
over-pessimistic reading caused by a noisy reference; this corrects an
over-optimistic one caused by uncontrolled clocks. A difference that hides inside
noise is not thereby absent — "below resolution" is a statement about the
harness, not about the engines, and the fix is a better harness, which is exactly
what produced a non-overlapping 2.5%.

**Row verdict: NOT parity on this cell. 0.975x, real and reproducible.**

## 6t. WHERE THE 2.5% LIVES: the MoE expert GEMM (2026-08-12)

nsys on our verify at T=9 (35B, k=8), `--cuda-graph-trace=node`, two token
lengths so the one-time prefill differences out.

**First, what it is NOT.** The 96-token profile's top entries look alarming:

| kernel | share (96-tok run) | instances |
|---|---|---|
| `TransposeToInt32Kernel` | 16.9% | 20561 |
| `gptq_marlin_repack_kernel` | 15.8% | 20561 |
| `ProcessScalesKernel` | 7.7% | 20561 |

40.4% of GPU time in weight REPACKING. But the 32-token run reports the SAME
20561 instances and the same totals (183.5 / 170.4 / 83.9 ms vs 183.0 / 170.2 /
83.5). Identical counts across run lengths means these are **LOAD-TIME**, not
per-step -- the nsys-aggregation trap this row has already been caught by once
("per-step Marlin repack" that was load-time). They cost ~437 ms once at startup
and nothing per token. Do not optimise them for decode.

**What actually scales is the expert GEMM:**

| quantity | value |
|---|---|
| `marlin_moe_wna16::Marlin` instances | 560 (32 tok) -> 1520 (96 tok) = ~15/token |
| GPU time | (249.2 - 90.9) ms / 64 tok = **2.47 ms/token** |
| share of wall at 135.9 tok/s (7.36 ms/token) | **~34%** |

That is the per-token cost, it is a GEMM, and being compute-bound it explains
§6s's clock sensitivity exactly: pinning the SM clock hurt whichever engine does
more SM work per token, and it hurt us more.

**To close the measured 2.5% end-to-end requires ~7% off this kernel** (2.5 / 34).

**Next step, and the rule that governs it:** profile the ORACLE's expert path and
pair kernels BY CALL COUNT before touching ours. This row has already learned
that the hard way -- a previous campaign found Marlin at 55.5% of our step and
concluded it was the gap, until upstream's own profile showed it at 57.2% of ITS
step, i.e. never the gap at all. A 34% share proves where our time goes, NOT that
upstream spends less there. Until both profiles exist, "optimise the expert GEMM"
is a guess.

## 6u. PAIRED PROFILE: the gap is 8.2% INSIDE the same Marlin MoE kernel (2026-08-12)

§6t located our per-token cost in the expert GEMM but refused to act on a share
alone, because this repo has been burned by exactly that. Here is the other half,
upstream profiled through its own torch profiler (nsys breaks its EngineCore),
warm generate only, 96 tokens, same model + draft + k.

**Paired BY CALL COUNT:**

| | kernel | instances | GPU time |
|---|---|---|---|
| ours | `marlin_moe_wna16::Marlin` | **1520** | **249.22 ms** |
| upstream | `marlin_moe_wna16::Marlin` | **1520** | **230.39 ms** |

The SAME kernel with the SAME template arguments and the SAME 1520 launches, and
we are **8.2% slower inside it**. Shares agree too: excluding our load-time
repack (437 ms, §6t) it is 38.8% of our decode-relevant GPU time against their
36.9%.

**The arithmetic closes:** the expert GEMM is ~34% of our wall (§6t), so 8.2%
slower there is 0.34 x 8.2 = **2.8% end-to-end**, against the **2.5%** measured
under pinned clocks (§6s). The whole residual is this one kernel; nothing else
needs explaining.

**This is the outcome the pairing rule exists to produce.** A previous campaign
found Marlin at 55.5% of our step, called it the gap, and upstream's profile then
showed 57.2% of ITS step -- a share proving only where time goes. Here the shares
are ALSO near-equal (38.8 vs 36.9), so share reasoning would again have said
"not the gap". Only the ABSOLUTE time at matched call count exposes it: identical
work, 18.8 ms more of it.

**What it is NOT:** not a different algorithm, not a different kernel family, not
launch overhead, not acceptance, not the repack (load-time), not the sampler
(bandwidth bound), not the graphs (both capture). Upstream reaches the same
kernel through `gptq_marlin_repack` at load exactly as we do.

**Next, and it is narrow:** the difference must be in how the kernel is
CONFIGURED or fed -- tile/thread selection (Marlin picks thread_k/thread_n per
shape), the grouped-GEMM problem layout at T=9, or the scales/zero-point layout
the repack produces. The template arguments printed in both profiles match, so
compare the LAUNCH parameters and the repacked weight layout, not the algorithm.
An 8.2% recovery on that kernel is worth the full 2.5%.

## 6v. INSIDE THE KERNEL: what the 8.2% is NOT (2026-08-12)

§6u paired the profiles and put the whole residual in `marlin_moe_wna16::Marlin`.
This narrows it further, and the cross-check below is what makes the finding
trustworthy in the first place.

**The instrumentation objection, answered.** Our number came from nsys and
upstream's from its torch profiler, so a sceptic should ask whether the tools
differ. Split the profiles:

| | ours | upstream | delta |
|---|---|---|---|
| Marlin MoE | 249.2 ms | 230.4 ms | **+8.2%** |
| EVERYTHING ELSE | 392.8 ms | 393.6 ms | **-0.2%** |
| total decode GPU | 642.0 ms | 624.0 ms | +2.9% |

If nsys inflated our kernels, every kernel would inflate. All other GPU work
matches to 0.2%, and the total (+2.9%) tracks the independently measured
pinned-clock wall gap (2.5%). The difference is real and confined to one kernel.

**Eliminated, by reading both sources:**

1. **Kernel selection.** The full template arguments are identical on both sides:
   `<...128, 1, 8, 4, true, 4, 1, false>`. Same instantiation, same tiling.
2. **The `block_size_m` clamp.** Upstream clamps to >= 16 for 1-byte input dtypes
   (`marlin_moe.py:328-329`) and our selector deliberately omits it. Tempting,
   but the identical template arguments prove both land on the same config here.
3. **Grid selection.** `determine_exec_config` is BYTE-IDENTICAL between our
   vendored copy and the pinned `csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu`
   (69 lines, no diff), and both callers pass `thread_k = thread_n = -1`, so both
   take the same auto path. The `blocks_per_sm` argument differs in spelling only
   (we pass 0, upstream defaults -1) and neither reaches the manual branch.
4. **`ignore_invalid_experts`.** Upstream passes it, but it only changes counting
   when an `expert_map` exists (expert parallelism). Single GPU has none.

**So: same kernel, same instantiation, same grid rule, same launch count, same
alignment semantics -- and 18.8 ms more work.** That leaves what the kernel READS:

- the repacked WEIGHT LAYOUT. Our load path runs two kernels upstream does not
  (`TransposeToInt32Kernel`, `ProcessScalesKernel`, §6t) to adapt a different
  source layout. Correctness is settled, but stride/padding/alignment of the
  final B and scales tensors is not, and the kernel's loads are sensitive to it.
- WEIGHT RESIDENCY. This repo has a standing GB10 finding that host/ATS-retagged
  weights run materially slower per GEMM than device-resident ones. Whether every
  35B expert tensor is device-resident on this path is not established here.

Both are checkable without guessing: dump the B/scales tensor strides and the
pointer residency on both sides for one expert, and compare. That is the next
step, and it is measurement, not a rewrite.

**Do not "optimise the kernel".** It is vendored from vLLM and byte-identical in
the parts that choose what to run; the difference is in what we hand it.

## 6w. THE INPUTS MATCH -- so the 8.2% may be ROUTING, not a defect (2026-08-12)

§6v narrowed the residual to "what we hand the kernel". Comparing that directly
closes off the layout hypotheses and opens a different, more likely one.

**Everything about the inputs matches:**

| property | ours | upstream |
|---|---|---|
| w13 scales per expert | `2*(K/16)*N` = 131072 B | `[256,128,1024]` fp8 = 131072 B |
| w2 scales per expert | `(N/16)*K` = 65536 B | `[256,32,2048]` fp8 = 65536 B |
| pointer alignment | pool over `cudaMalloc` | `ptr%16 == 0`, `ptr%256 == 0` |
| residency | `cudaMalloc` (device) | torch CUDA tensor (device) |

So: identical kernel, identical instantiation, identical grid rule, identical
launch count, identical scale layout, identical alignment, identical residency.
At that point "our kernel is slower" stops being the simplest explanation.

**The likelier one: we are not doing the same WORK.** The Marlin MoE kernel loops
`div_ceil(num_tokens_past_padded, moe_block_size)` blocks per launch, and
`num_tokens_past_padded` depends on HOW MANY DISTINCT EXPERTS the batch touches:
E=256, top_k=8, M=9 gives up to 72 (token, expert) pairs, each padded to a
`block_size_m` multiple. The launch COUNT is fixed by layers x steps (1520 on
both sides, as measured), but the BLOCKS PER LAUNCH are not.

And our token stream is NOT upstream's. §6i measured our 35B "fibonacci" output
diverging from the oracle's at char 55 (`return (` vs `return(`) -- a near-tie
flip, the ratified regime. Different tokens route to different experts, so the
two engines can execute different amounts of expert work for the "same" prompt
while launching the kernel the same number of times.

**If that is the cause, the 8.2% is not an implementation gap and cannot be
optimised away** -- it is the cost of a different (equally valid) token path.

**The experiment that decides it**, and it must run before any kernel work:
instrument `num_tokens_past_padded` (or the per-call block count) on both sides
for the same prompt and compare the TOTALS over a run. Equal totals => our kernel
really is slower and the layout hunt continues elsewhere. Different totals => the
gap is routing, the comparison was never like-for-like at the kernel level, and
the honest statement is that the engines diverge in what they compute rather than
how fast they compute it.

Recording this BEFORE acting, because every cheap explanation has now been
eliminated and the expensive one (rewrite the expert path) would be exactly the
wrong response to a routing difference.

## 6x. ROUTING REFUTED: our kernel is 12.8% slower PER UNIT OF WORK (2026-08-12)

§6w proposed that the 8.2% Marlin gap might not be a defect at all -- that our
divergent token stream routes to different experts, so the two engines execute
different amounts of expert work at the same launch count. Measured on both
sides (`VT_MOE_PAD_STATS=1` here; `moe_align_block_size` wrapped upstream), same
prompt, same k:

| | avg padded tokens / call | avg blocks / call | block size |
|---|---|---|---|
| ours | 311.2 | **38.9** | 8 |
| upstream | 324.8 | **40.6** | 8 |

**Upstream loops 4.4% MORE blocks per launch than we do, and is still 8.2%
faster.** The hypothesis is refuted, and it was protective: normalising time by
the work actually performed makes our deficit BIGGER, not smaller.

| | time | launches | blocks/launch | **per block** |
|---|---|---|---|---|
| ours | 249.2 ms | 1520 | 38.9 | **4.21 us** |
| upstream | 230.4 ms | 1520 | 40.6 | **3.73 us** |

**Our Marlin MoE is ~12.8% slower per unit of work.** Both also choose block_size 8,
which independently confirms §6v's conclusion that upstream's `>= 16` clamp does
not bite on this shape.

So the gap is a REAL in-kernel execution difference, now quantified per block and
with every input-side explanation eliminated (§6v, §6w): same kernel, same
instantiation, same grid rule, same scale layout, same alignment, same residency,
and now MORE work on their side. What remains is how the kernel executes given
identical inputs -- occupancy, shared-memory budget, or the `max_shared_mem`
value each side passes, which is the one launch input not yet compared.

**Two diagnostic traps recorded**, both hit while measuring this:

1. Our probe read a device value inside the verify, which W8 now CAPTURES:
   `cudaStreamSynchronize: operation not permitted when stream is capturing`.
   Our own capture work made the instrument illegal in that region; the count is
   capture-independent, so it runs with `VT_SPEC_DECODE_GRAPH=0`.
2. Upstream's wrapper sat inside a `torch.compile` region and broke compilation;
   the count is compile-independent, so it runs under `enforce_eager=True`.

Neither changes what is counted, and saying so is the point: a work COUNT may be
taken under different execution modes, a TIME may not.

## 6y. EVERY SOURCE-LEVEL EXPLANATION IS EXHAUSTED (2026-08-12)

§6x quantified the residual at 12.8% per unit of work inside `marlin_moe_wna16`.
This is the complete elimination list, so the next person does not redo it:

| checked | ours | upstream | verdict |
|---|---|---|---|
| kernel source | vendored `marlin_mm_moe.cu` | `csrc/.../ops.cu` | **dispatcher VERBATIM**; our only additions are includes and a default-OFF E=1 clamp that cannot fire for MoE |
| template instantiation | `<...128, 1, 8, 4, true, 4, 1, false>` | identical | same tiling, same a/b/c/s scalar types |
| `determine_exec_config` | 69 lines | identical | byte-identical, both callers take the auto path |
| `moe_block_size` | 8 | 8 | measured on both; upstream's `>= 16` clamp does not fire |
| `max_shared_mem` | `cudaDeviceGetAttribute` + `/blocks_per_sm - 1024` | identical | same |
| `use_atomic_add` / `use_fp32_reduce` / `is_k_full` | false / true / true | false / true / true | same |
| scale bytes per expert | 131072 / 65536 | 131072 / 65536 | same |
| pointer alignment | pool over `cudaMalloc` | `ptr%256 == 0` | same |
| residency | `cudaMalloc` (device) | torch CUDA tensor | same |
| WORK per launch | 38.9 blocks | 40.6 blocks | upstream does **MORE** |
| CUDA toolkit | 13.0 (V13.0.88) | torch 2.13.0+cu130 | same |
| arch / flags | `121a`, `-O3 -DNDEBUG` | sm_121 kernels in trace | same |

Identical code, identical parameters, identical toolchain, identical arch, and
LESS work on our side -- still 4.21 us/block against 3.73.

**So the answer is not in the source, and further source reading is waste.** What
remains is how the two binaries actually execute: achieved occupancy, register
count, L2 hit rate, memory throughput and stall reasons. That is an `ncu`
question, and it is the next step:

    sudo ncu --set full --kernel-name-base mangled \
             --kernel-name regex:marlin_moe_wna16 --launch-count 20 <cmd>

on BOTH engines (passwordless `sudo ncu` is available on this box), comparing
`sm__throughput`, `achieved_occupancy`, `launch__registers_per_thread` and the
top stall reason. Equal occupancy with different memory throughput points at
data placement; different occupancy points at register pressure, i.e. a codegen
difference between two builds of the same source.

**Do not start by editing the kernel.** Everything editable has been shown
identical; the difference is in the compiled artifact or its runtime conditions,
and only a profiler at that level can say which.

## 6z. THE BINARIES ARE EQUIVALENT: the residual is RUNTIME, not code (2026-08-12)

§6y said the answer was no longer in the source and named `ncu` as the next step.
Ran it, plus a static comparison of the shipped machine code. Both close.

**Measured on our side (`ncu`, 10 launches, steady state):**

| metric | ours |
|---|---|
| registers per thread | **94** |
| warps active (occupancy proxy) | 24.0% |
| SM throughput | 10.6% of peak |
| L1 sector hit rate | 0.7% |
| L2 sector hit rate | 9.5% |

A latency-bound kernel, as expected for a W4A16 grouped GEMM at M=9.

**Static comparison of the SAME instantiation** (`<...2814749767172868, 128, 1, 8,
4, true, 4, 1, false>`), ours from our build object, upstream from the shipped
`_moe_C_stable_libtorch.abi3.so`:

| | registers | SASS instructions |
|---|---|---|
| ours (sm_121a) | **94** | **3664** |
| upstream (sm_120) | **94** | **3664** |

**The compiled kernels are equivalent.** Note upstream ships no sm_121 cubin at
all -- sm_80/87/89/90/90a/100/110/120 -- so on GB10 it runs the family-compatible
sm_120 code while we compile sm_121a, and the two compile to the same instruction
count with the same register pressure.

**So every code-level explanation is now eliminated**, including the machine code
itself: source, instantiation, grid, block size, shared memory, reduction flags,
scale layout, alignment, residency, toolkit, arch, register pressure and SASS
length all match, and upstream performs MORE work per launch (40.6 blocks vs
38.9). Ours still takes 4.21 us/block against 3.73.

**The residual is therefore RUNTIME, not code.** What differs is the environment
the identical kernel executes in: how each engine's allocator places the expert
weights and activations, and hence L2/DRAM locality across the 256 experts. Our
L2 hit rate is 9.5%, so this kernel lives or dies on memory placement, and the
two engines allocate differently (our `DevicePool` slabs vs torch's caching
allocator) even though both are `cudaMalloc`-backed, contiguous per tensor and
256-byte aligned.

**What is NOT worth doing:** editing the kernel, its launch config, its layout or
its build flags. All are proven identical. This row has now spent a full
investigation arriving at that, and the value of recording it is that nobody
repeats it.

**The one experiment left** needs upstream's `ncu` counters for the same kernel,
which is currently BLOCKED: vLLM's EngineCore fails to initialise under `ncu`'s
kernel replay ("Engine core initialization failed"), so the comparison of
occupancy / L2 hit rate / DRAM throughput cannot be completed the same way.
Options are `--replay-mode application`, profiling a smaller standalone harness
that calls the op directly, or accepting that the last 2.5% is unattributed.

**Row verdict unchanged and now final for this campaign: 0.975x on the code cell,
1.012x on the prose cell, NOT parity, residual real and unattributed.**

## 6aa. FINAL ATTRIBUTION: a 12.9% EFFECTIVE-BANDWIDTH gap on identical code (2026-08-12)

§6z showed the machine code is equivalent and called the residual "runtime,
unattributed". It can be attributed further, because a memory-bound kernel's
time IS its achieved bandwidth.

The 35B-A3B expert weights are 4-bit: gate_up `K*2N/2` = 1.05 MB per expert,
down `N*K/2` = 0.52 MB (K=2048, N=512), so the average launch reads 0.79 MB per
expert-block. With the measured blocks per launch (§6x) and time per launch:

| | us / launch | MB / launch | **effective bandwidth** |
|---|---|---|---|
| ours | 164.0 | 30.59 | **186.6 GB/s** |
| upstream | 151.6 | 31.93 | **210.7 GB/s** |

**Upstream sustains 12.9% more effective DRAM bandwidth**, which is the whole
per-unit-work gap (12.8%) to within rounding. The kernel is DRAM-bound (L2 hit
rate 9.5%, SM throughput 10.6%), so this is not a coincidence -- for this kernel,
time and achieved bandwidth are the same measurement.

**Residency is not the cause.** This repo has a standing GB10 finding that
host/ATS-retagged weights run materially slower per GEMM, with staging-at-load as
the fix. That fix is ALREADY applied here: `ResidentWeight`
(`dense_attn_block.h:190-196`) and `BuildMoeMarlinResident` both allocate through
`d.b.Alloc` -> `cudaMalloc` and upload once, so the expert weights are true
device memory, contiguous and 256-byte aligned, exactly like upstream's tensors.

**So the honest final statement:** identical machine code, identical launch
parameters, identical weight layout and residency, LESS work on our side, and
13% lower achieved bandwidth reading the same bytes. The cause is in the memory
system's behaviour for our allocation (physical page backing / TLB coverage of a
single ~512 MB slab spanning all 256 experts, versus torch's segmented caching
allocator), not in anything the kernel or its inputs express.

**Next levers, in order of cheapness:**
1. ~~Split the per-expert weight slab~~ -- **REFUTED, do not try it.** Our slab is
   the SAME SIZE and stride as upstream's tensor: per-expert gate_up slot
   `2*wg_i32` = 262144 int32 = exactly the 1.0 MB a `[K, 2N]` 4-bit weight needs
   (no padding), and the whole slab is 268 MB on both sides. There is no
   oversizing or stride inflation to remove.
2. `cudaMemAdvise` / preferred-location hints on the expert slab.
3. Upstream's `ncu` counters -- **BLOCKED, both replay modes TRIED.** Its
   EngineCore fails to initialise under `ncu` with the default kernel replay AND
   with `--replay-mode application` (identical "Engine core initialization
   failed"), so the profiler's mere presence breaks vLLM's init on this stack.
   Its DRAM efficiency therefore stays DERIVED (210.7 GB/s from time x bytes),
   not directly counted. A standalone harness that calls `moe_wna16_marlin_gemm`
   outside the engine is the only remaining route.

**Campaign verdict: 0.975x code cell, 1.012x prose cell, NOT parity.** The
residual is now measured, localised to one kernel, quantified per unit of work,
and attributed to achieved memory bandwidth rather than left as "unexplained".

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
