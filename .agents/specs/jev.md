# SPEC — `MODEL-JEV`: Jev-like structured generation for DiffusionGemma (vLLM PR #57250)

Port vLLM PR #57250 into vllm.cpp, adding a structured generation mode for
DiffusionGemma that enables Jev-like bounded-choice answers with canvas
seeding, read-only requests, pinned positions, and logprobs on the converging
step. DiffusionGemma is a 26B-parameter (4B active) MoE discrete diffusion
language model built on the Gemma4 backbone, already inventoried in vllm.cpp.
This row adds the decision capability: the model can answer noul (yes/no),
scale, and multiple-choice questions with certainty and error bars. CPU + GPU
(CUDA), served through the existing `/v1/systemone` API surface.

## Now

`SPEC`

## Scope

- **Row.** `MODEL-JEV` (this spec). New model-matrix row that ports vLLM PR
  #57250's structured generation mode for DiffusionGemma, enabling Jev-like
  bounded-choice decisions on top of the already-inventoried DiffusionGemma
  backbone.
- **In.** The structured generation mode: canvas seeding
  (`diffusion_seed_canvas`), read-only requests (`diffusion_read_only`),
  step cap (`diffusion_max_steps`), pinned seeded-canvas positions through
  denoising steps, per-request canvas width, sampler tiling by canvas width,
  temperature-1 logprobs on the converging step, request validation for
  diffusion `extra_args`, scheduler integration (step cap enforcement, canvas
  width tracking, async scheduling for narrower canvases), self-conditioning
  skip for one-step tiles, logprob stash joining across different canvas
  widths, and the example structured-decision server. The DiffusionGemma model
  itself (encoder/decoder dual-mode, self-conditioning MLP, diffusion sampler,
  canvas mechanism) if not already ported by the inventory. CPU build + tests;
  GPU forward (CUDA).
- **Out (owned by other rows).** The vLLM pin advance as a standalone sync
  cycle (Phase 1 does it, but the cycle itself follows `upstream-sync.md` and
  reconciles all affected rows, not just MODEL-JEV). ROCm kernel tuning. GGUF
  k-quants for DiffusionGemma (owed, named below). LocalAI backend integration
  is a separate PR in a separate repo. The base DiffusionGemma model port
  (encoder/decoder/sampler) is in scope ONLY if the inventory has not already
  ported it; if it has, this row adds the structured generation layer on top.
- **Reuse.** The Gemma4 backbone (dense + MoE) already ported in vllm.cpp
  (`gemma4.cpp`, `gemma4_moe.cpp`, `gemma4_weights.cpp`,
  `gemma4_registry.cpp`). The `/v1/systemone` API endpoints, request/response
  structs, and server dispatch from the Laya/kev work. The `vllm_decide` /
  `vllm_decide_free` C ABI at v29. The `DecisionFn` callback mechanism from
  Laya. The Gemma vocabulary, tokenizer, PLE, YOCO KV-sharing, and
  heterogeneous head_dim infrastructure already in the tree.

## Upstream chain

### Oracle: vLLM (PR #57250 merged to main)

Repository: `vllm-project/vllm`. PR #57250 head at
`a7c23ac96d7806e7c7e7d862eadbce5a33529b94`, merge-base
`5559679229bc961848b121ccdeaa8fa5d79bec98` (the prior vllm.cpp parity pin).

The current vllm.cpp parity pin is `e126687a9a828d513c01a07cd69f025f27d63280`
(0.28.1rc1.dev132). PR #57250 is AHEAD of this pin — its merge-base is the
prior pin `5559679229`. **The vLLM pin must be advanced past PR #57250 before
the structured generation mode can be ported.** This advance is Phase 1.

Key upstream source files (at PR #57250 head):

- `vllm/model_executor/models/diffusion_gemma.py` — the DiffusionGemma model,
  `ModelState`, `Sampler`. Single Gemma4 backbone run in two modes: encoder
  mode (causal attention, writes KV cache) and decoder mode (bidirectional
  attention, reads encoder KV, does not write). Same weights, same layers. The
  only decoder-unique component is a self-conditioning MLP.
- `vllm/transformers_utils/configs/diffusion_gemma.py` — DiffusionGemma config
  (`DiffusionGemmaConfig`).
- Core scheduler / request paths — request validation for diffusion
  `extra_args`, per-request canvas width, step cap enforcement, read-only
  request lifecycle, logprob emission at convergence.
- `examples/structured_diffusion/` — the example structured-decision server
  for DiffusionGemma reads (the Jev-like bounded-choice application).

PR #57250 adds three fields to `SamplingParams.extra_args`:

- `diffusion_seed_canvas` — replaces the random initialization after prefill
  with a caller-supplied token ID sequence exactly the length of the canvas.
  Seeded positions are **pinned** through all denoising steps (commit
  `408f00d784`): the sampler does not denoise them.
- `diffusion_max_steps` — the upper limit for the number of denoising steps.
  The step cap is kept in the step counter's dtype (commit `21562ccfe1`).
- `diffusion_read_only` — outputs an argmax canvas upon convergence, skips
  further generation, and reports temperature-1 logprobs on the converging
  step (commit `fa81b63406`). Read-only requests end on the canvas they emit
  (commit `15eb9be76f`).

Supporting changes in the PR:

- Per-request canvas width for diffusion reads (commit `5b6f205d8e`).
- Tile the sampler by canvas width (commit `9e9a23f21d`).
- Keep canvas widths in the scheduler (commit `55f03bb5c8`).
- Require async scheduling for narrower diffusion canvases (commit
  `2449995bf8`).
- Do not schedule a step past a read-only request's cap (commit
  `8dc04ec6f3`).
- Skip self-conditioning for one-step tiles (commit `0c64f9fedb`).
- Join logprob stashes of different widths (commit `d9af863210`).
- Keep the sampler step compiled and correct in eager (commit `a68dab46ca`).
- Validate diffusion `extra_args` whenever the model is a diffusion model
  (commit `7ba6675684`).
- Isolate diffusion request validation and simplify read emission (commit
  `dd8639536d`).
- Register the top-level model type `diffusion_gemma` with the Gemma4 arch
  convertor (commit `a7c23ac96d`).

### Base model: DiffusionGemma

- `google/diffusiongemma-26B-A4B-it` — 26B-parameter (4B active) Mixture-of-
  Experts discrete diffusion language model. Built on the Gemma4 backbone (MoE
  variant). Uses Uniform State Diffusion: replaces text with random vocabulary
  noise and iteratively refines a fixed-size canvas of `canvas_length` tokens
  in parallel. Encoder (causal) reads the clean prompt into a KV cache; decoder
  (bidirectional) denoises the canvas by cross-attending to that cache.
  Generation alternates an outer autoregressive loop over canvases with an
  inner denoising loop.
- The Gemma4 backbone (PLE, YOCO KV-sharing, heterogeneous head_dim,
  proportional partial-RoPE, GeGLU MLP, MoE router) is already fully implemented
  in vllm.cpp: `gemma4.cpp`, `gemma4_moe.cpp`, `gemma4_weights.cpp`.

### Structured-decision application (Jev-like)

The example server in `examples/structured_diffusion/` (PR #57250) applies the
structured generation mode to produce Jev-like bounded-choice answers. The
model answers:

- **Noul** (yes/no): seed the canvas with the question + two choice tokens,
  run read-only diffusion, read the argmax + logprobs at convergence.
- **Scale** (ordinal scale): seed the canvas with the question + level tokens,
  read the convergent distribution for certainty and error bars.
- **Multiple-choice**: seed the canvas with the question + option tokens, read
  the convergent distribution.

Certainty and error bars are derived from the temperature-1 logprobs on the
converging step.

## Design

### Phase 1: Advance the vLLM pin past PR #57250

The current parity pin `e126687a9a` is BEHIND PR #57250. The pin must be
advanced to at least `a7c23ac96d` (PR #57250 head) before any structured
generation code can be ported.

Follow the sync cycle in `.agents/upstream-sync.md`:

1. Fetch `origin/main` in the reference checkout. Target = a commit at or past
   PR #57250 head.
2. Enumerate commits from the current pin to target across all mirrored
   subtrees.
3. Classify every commit: PORT-NOW, INVENTORY, IGNORE.
4. Write the sync report to `.agents/sync/YYYY-MM-DD-<target7>.md`.
5. Port the PORT-NOW queue. Bump file headers. Append `parity-ledger.md` rows.
6. Re-verify: regenerate goldens at target, run op/behavioral/model suites.
7. Advance the pin. Update the `parity-pin` block in `upstream-sync.md` and
   the oracle block in `oracles/vllm.md`.

**This phase reconciles every affected row and gate, not just MODEL-JEV.**

### Phase 2: Understand the structured generation mode

Read the upstream source at PR #57250 head and document:

- The DiffusionGemma model architecture: encoder/decoder dual-mode, KV cache
  sharing, self-conditioning MLP, canvas mechanism, diffusion sampler.
- The three `extra_args` fields and their exact semantics.
- The scheduler changes: per-request canvas width, async scheduling, step cap
  enforcement, logprob stash joining.
- The sampler changes: tiling by canvas width, self-conditioning skip,
  compiled vs eager correctness.
- The request validation.
- The example structured-decision server: how it maps noul/scale/multiple-
  choice questions to seeded canvases, how certainty and error bars are
  computed.

This phase produces no code. It produces a design document for Phase 3.

### Phase 3: Port the structured generation mode to vllm.cpp

Port the upstream changes into the mirrored C++ files:

- **DiffusionGemma model**: port the model (encoder/decoder modes, self-
  conditioning MLP, diffusion sampler, canvas mechanism). Map
  `diffusion_gemma.py` to `src/vllm/model_executor/models/diffusion_gemma.cpp`
  + `include/vllm/model_executor/models/diffusion_gemma.h`.
- **Canvas seeding**: port `diffusion_seed_canvas` — caller-supplied token ID
  sequence replaces random init. Port position pinning through denoising steps.
- **Read-only requests**: port `diffusion_read_only` — argmax canvas upon
  convergence, skip further generation, report temperature-1 logprobs. Port
  the lifecycle: read-only requests end on the canvas they emit.
- **Step cap**: port `diffusion_max_steps` — upper limit on denoising steps.
  Port the scheduler guard: do not schedule past a read-only request's cap.
- **Per-request canvas width**: port per-request canvas width, canvas width
  tracking in scheduler, async scheduling requirement for narrower canvases.
- **Sampler tiling**: port tiling by canvas width, self-conditioning skip for
  one-step tiles, logprob stash joining, compiled/eager correctness.
- **Request validation**: port diffusion `extra_args` validation.
- **Top-level model type**: port registration of `diffusion_gemma` with the
  Gemma4 arch convertor.

### Phase 4: Registration + serving

- Register DiffusionGemma via `REGISTER_VLLM_MODEL` in a new
  `diffusion_gemma_registry.cpp`. The model type `diffusion_gemma` maps to the
  Gemma4 backbone factory with diffusion-specific extensions.
- Wire the structured generation mode into the `/v1/systemone` dispatch. The
  `vllm_decide` ABI (v29) carries the request JSON; the engine detects the
  DiffusionGemma architecture and runs the structured generation pipeline
  (seed canvas, denoise, read argmax + logprobs at convergence).
- Map the Jev-like bounded-choice question types to the existing
  `/v1/systemone` API:
  - **Noul**: seed canvas with question + yes/no tokens, read convergent
    `p(true)`.
  - **Scale**: seed canvas with question + level tokens, read convergent
    distribution for score + certainty.
  - **Multiple-choice**: seed canvas with question + option tokens, read
    convergent distribution for choice + confidence.
- Certainty and error bars computed from temperature-1 logprobs on the
  converging step, matching the example structured-decision server's formulas.
- All probabilities rounded to 2 decimals (same as kev/Laya).

### Phase 5: GPU (CUDA)

- GPU build verification: DiffusionGemma (26B A4B MoE) forward on CUDA. The
  Gemma4 MoE CUDA path already exists; the diffusion-specific additions
  (bidirectional attention, self-conditioning MLP, sampler tiling) must build
  and run on device.
- E2E on GPU: structured generation through the registered forward path.
- Not a pre-PR gate (stop condition is correct on CPU).

## Our baseline

Before this row: the Gemma4 backbone (dense + MoE) is fully ported in vllm.cpp
with PLE, YOCO KV-sharing, heterogeneous head_dim, proportional partial-RoPE,
and MoE router support. DiffusionGemma is inventoried but the diffusion model
(encoder/decoder dual-mode, self-conditioning MLP, diffusion sampler, canvas
mechanism) and the structured generation mode are NOT ported. The
`/v1/systemone` API and C ABI `vllm_decide` exist at ABI v29 from the Laya/kev
work. The current vLLM parity pin (`e126687a9a`) is BEHIND PR #57250.

## Port map

- DiffusionGemma model: vLLM `diffusion_gemma.py` @ PR #57250 head →
  `src/vllm/model_executor/models/diffusion_gemma.cpp` +
  `include/vllm/model_executor/models/diffusion_gemma.h` (new files).
- DiffusionGemma config: vLLM `diffusion_gemma.py` config → C++ config loader.
- Structured generation `extra_args`: vLLM `sampling_params.py` → C++ engine's
  request/sampling params struct + validation.
- Scheduler changes: vLLM `vllm/v1/` scheduler → C++ engine scheduler.
- Sampler changes: vLLM `diffusion_gemma.py` sampler →
  `src/vllm/model_executor/models/diffusion_gemma.cpp`.
- Top-level model type registration: vLLM arch convertor → C++ model registry.
- Registration: `src/vllm/model_executor/models/diffusion_gemma_registry.cpp`
  (new file, self-registers via `REGISTER_VLLM_MODEL`).
- SystemOne dispatch: shared `src/vllm/entrypoints/openai/systemone.{h,cpp}`
  (reused from Laya/kev).
- C ABI: `vllm_decide` / `vllm_decide_free` in `include/vllm.h` (ABI v29;
  no new ABI needed). The architecture allowlist in `vllm_decide`
  (`src/capi/vllm_c.cpp:1711-1718`) currently accepts `KevModel`,
  `LayaModel`, `CuaS1Forms`. Add `"DiffusionGemma"` to this list and
  add a diffusion-gemma structured-generation branch so it is dispatched
  by engine architecture name — same pattern, same ABI, one more family.
  Similarly, `server_main.cpp` must route the `"DiffusionGemma"`
  architecture to the diffusion decision callback via `set_decision`.
- Tests: `tests/vllm/models/test_diffusion_gemma.cpp` (new file).

## Tests to port

Port the upstream vLLM tests for PR #57250's structured generation mode:

- Canvas seeding: verify `diffusion_seed_canvas` replaces random init, seeded
  positions pinned through all denoising steps.
- Read-only requests: verify argmax upon convergence, skip further generation,
  temperature-1 logprobs on converging step, end on emitted canvas.
- Step cap: verify `diffusion_max_steps` caps denoising, scheduler does not
  schedule past cap.
- Per-request canvas width: verify narrower canvases, scheduler tracking, async
  scheduling.
- Sampler tiling: verify tiling, self-conditioning skip, logprob stash joining.
- Request validation: verify invalid `extra_args` rejected.
- Top-level model type: verify `diffusion_gemma` registers with Gemma4 arch
  convertor.
- E2E structured decisions: noul, scale, multiple-choice — verify certainty
  and error bars from temperature-1 logprobs.

## Dependencies

- The vLLM pin advance past PR #57250 (Phase 1) — prerequisite for all other
  phases.
- The Gemma4 backbone (dense + MoE) already ported.
- The `/v1/systemone` API and C ABI `vllm_decide` (ABI v29).
- No new CUDA kernels expected — routes through existing `vt::` ops and Gemma4
  MoE CUDA path.

## Work breakdown

- Phase 1: Advance vLLM pin past PR #57250 — TODO.
- Phase 2: Understand structured generation mode — TODO.
- Phase 3: Port structured generation mode to vllm.cpp — TODO.
- Phase 4: Registration + serving — TODO.
- Phase 5: GPU (CUDA) — TODO.

## Risks

- **Advancing the vLLM pin is itself a risk.** The advance from `e126687a9a`
  to PR #57250 head must reconcile every affected row and gate. Existing owed
  obligations at the current pin compound this risk. A stalled sync cycle keeps
  the old pin; MODEL-JEV cannot proceed past Phase 2 until the pin is advanced.
- **DiffusionGemma model port scope.** If the inventory has not already ported
  the DiffusionGemma model, Phase 3 must port the full model in addition to
  the structured generation mode. Phase 2 determines this.
- **Bidirectional attention.** The decoder mode uses bidirectional attention,
  not the standard causal path. The existing vllm.cpp attention infrastructure
  may need a new attention mask mode.
- **MoE memory.** DiffusionGemma is 26B (4B active). CPU forward may be
  impractical for full-model E2E tests; reduced fixtures may be needed.
- **Async scheduling requirement.** Narrower diffusion canvases require async
  scheduling. If vllm.cpp does not yet support this, it is a dependency that
  must be resolved before Phase 3.
- **Logprob stash widths.** The C++ port must handle variable canvas widths
  in the logprob path from the start.

## Gates

- CPU-correct: all ported tests pass.
- E2E parity: noul/scale/multiple-choice outputs match upstream example
  structured-decision server within tolerance.
- Reachability: `/v1/systemone` serves DiffusionGemma structured decisions via
  `vllm_decide`.
- GPU build verification: owed.

## Stop conditions

- CPU-correct + E2E parity + reachability = ready for PR.
- GPU build = owed.
- GGUF k-quants = owed.
- Phase 1 (pin advance) must complete before Phase 3 begins.

## Git integration

One pull request (repository default policy). Spec commit precedes
implementation commits in the same pull request. Phase 1 (pin advance) may be
a separate PR if the sync cycle is large enough — ask the developer at row
claim per AGENTS.md §115.

## Owed

- GPU (CUDA) build verification + E2E.
- GGUF k-quant arm for DiffusionGemma.
- Determination of whether the DiffusionGemma model is already ported by the
  inventory or must be ported as part of this row (Phase 2 resolves this).
- The vLLM pin advance past PR #57250 (Phase 1) — prerequisite; if split into a
  separate PR, must land before structured generation code.
