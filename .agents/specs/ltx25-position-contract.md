# `LTX25-POSITION-CONTRACT` — the tower positions get an integer gate, and the note that claimed a value gate stops

Issue: [#1467](https://github.com/mudler/vllm.cpp/issues/1467).
Owning row: `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model`, which
carries #1467 under `## Owed` in [`ltx-2-5.md`](ltx-2-5.md).

This row has **no matrix row and therefore no lifecycle state**. It repairs one
instrument and one stale note inside a shipped path; its state is the issue it
closes and the test that holds it.

## Scope

IN SCOPE:

1. **The layer-two question #1467 raised and did not answer.** Its table shows
   the position-renumbering MUTANT scoring closer to the oracle than the port.
   The natural reading is that our position handling is wrong. Establish whether
   it is, from upstream source at the pin and from the oracle re-run, before
   touching the note. `## The verdict`.
2. **The self-contradiction in `test_ltx2_text_encoder.cpp`.** The case note at
   `:2303-2307` claims renumbering "does red this case, but only at 1.10x the
   audio floor". The measured table 140 lines below it, in the same file, shows
   that claim is false post-`4712dac40`. Correct the note. Keep the table.
3. **An instrument that actually detects the renumbering.** #1467's own
   preferred repair: assert the integer `positions` vector the path builds,
   before any bf16 arithmetic can absorb it.

OUT OF SCOPE, each named because each was available:

- **Widening or moving any tolerance.** `2.0 * floor` on all four value arms is
  untouched, in form and in constant. [#1668](https://github.com/mudler/vllm.cpp/issues/1668)
  forbids exactly the move of recovering a detection by loosening a bound, and
  the detection here is not recoverable by any constant anyway — the ordering of
  correct and mutant has reversed, which no threshold can undo.
- **Deleting the measured table.** It is the evidence that the note was wrong.
- **Changing `4712dac40`'s rounding polarity.** It is right; #1458 established
  that it is the only form reproducing the `silu_and_mul_bf16_8x256` golden
  bit-exactly.
- **Rebuilding the rope-table case** (#1467's candidate 2). Candidate 1 is
  strictly upstream of it — the rope table cannot see a wrong position that the
  `positions` vector never carried — and shipping both would gate the same
  integer twice.
- **Any GPU leg.** None was authorised for this row and none is needed: every
  measurement here is CPU-only.

## The verdict on layer two: the port is RIGHT, and the mutant is NOT closer

#1467's table compares OUR bf16 conditioning against the ORACLE's bf16
conditioning, end to end, after 13 states are stacked and pushed through the
caption projections. That quantity cannot answer the question, and this row's
first job was to ask it somewhere it can be answered.

### What upstream does, read at the pin

Upstream never passes `position_ids` on this path, so transformers derives them,
and what it derives counts the pad rows:

| side | anchor | what it shows |
|---|---|---|
| LTX-2 `fd4ded7f` | `packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py:64-68` | `self.model.model(input_ids=..., attention_mask=..., output_hidden_states=True)` — the whole argument list. No `position_ids`. |
| LTX-2 `fd4ded7f` | `.../gemma/gemma_assets.py:162`, `.../gemma/encoders/base_encoder.py:231-236`, `.../gemma/tokenizer.py:25,50-51` | `TOKENIZER_MAX_LENGTH = 1024`, `PaddingSide.LEFT`, `padding="max_length"`. |
| diffusers `3a2f35d4` | `src/diffusers/pipelines/ltx2/pipeline_ltx2.py:347-349` | the same three arguments, read as an independent second opinion. `:329` sets `padding_side = "left"`. |
| transformers 5.12.1 | `models/gemma4_unified/modeling_gemma4_unified.py:1092-1096` and `:640-643` | `if position_ids is None: ... position_ids = torch.arange(inputs_embeds.shape[1], ...) + past_seen_tokens`. `inputs_embeds.shape[1]` is the PADDED length, `past_seen_tokens` is 0. The pads consume `0..npad-1` and the real tokens start at `npad`. |
| transformers 5.14.1 (the parity pin) | same file `:648-651` and `:1098-1101` | byte-identical branch. The two versions do not differ on this point. |
| transformers 5.12.1 | `generation/utils.py:713-727`, sole base call site `:2483` | the `attention_mask.long().cumsum(-1) - 1` renumbering exists — inside `generate()`. `Gemma4UnifiedForConditionalGeneration` does not override `_prepare_position_ids_for_generation`, and a plain `nn.Module.__call__` on `Gemma4UnifiedModel` never reaches it. |

So upstream's real tokens sit at absolute positions `first_valid .. first_valid+T-1`.
`ltx2_text_encoder.cpp:1100-1101` writes exactly that (`:1090-1091` when
#1467 cited it; this row's own comment additions moved it). The mutant does not.

vLLM defines nothing here, and that is stated rather than assumed: `grep -rn -i
"ltx"` over `vllm/model_executor/models/`, over `--include='*.py'` for the whole
`vllm` checkout at `5559679229`, `grep -rn -i "lightricks"`, and
`grep -rn "LTXVideo\|ltx_video\|LTX2\|LTX-2"` all return zero. It does ship
`vllm/model_executor/models/gemma4_unified.py` (registered at
`registry.py:405-407`) but as a generative multimodal model, never as a
left-padded text encoder, and V1 hands `positions` in per request as absolute
offsets. transformers is the executing chain for this question, and AGENTS.md's
`transformers` oracle row is what admits it.

### What the oracle answers when you ask it directly

Re-ran the committed generator's own tower — the same reduced Gemma-4 the
goldens came from, `transformers 5.12.1` / `torch 2.12.1+cpu`, CPU only, 168
parameters and buffers filled from the same `Ltx2Rand` stream — on three legs:

- `P`, the full LEFT-PADDED run, 12 pads then the 8 tokens, no explicit
  `position_ids`, i.e. what upstream runs;
- `ABS`, the 8 valid tokens alone at positions 12..19, i.e. what the port does;
- `ZERO`, the 8 valid tokens alone at positions 0..7, i.e. the mutant.

Per-state `max|.|` against `P`'s valid rows, worst state of 13:

| dtype | `|ABS - P|` | `|ZERO - P|` | states where ABS is closer | states where ZERO is closer |
|---|---:|---:|---:|---:|
| f32 | 5.257e-05 | 1.037e-04 | 10 of 13 | 2 of 13 |
| bf16 | **4.375e-01** | **1.156e+00** | **12 of 13** | **0 of 13** |

Re-run at the PARITY PIN's `transformers 5.14.1` (`/home/mudler/venvs/music3-oracle`,
`torch 2.11.0+cu130`, `CUDA_VISIBLE_DEVICES=""`) reproduces every cell of that
table byte-for-byte, so the reading is not a property of one transformers or one
torch.

`|P|` peaks at 14.35, so in f32 both are round-off: 3.66e-06 and 7.23e-06
relative. That is the physics, and it reproduces the number the production
comment already carried — `|ABS - ZERO|` in f32 measures 5.114e-05 here against
the `5.11e-05` recorded at `ltx2_text_encoder.cpp:1087`. Renumbering is exactly
a no-op in real arithmetic, because rotary embedding depends only on `m - n` and
the pads are masked out.

In bf16 it is not a no-op, and the direction is unambiguous: renumbering is
**2.64x further** from upstream's own bf16 answer, and it is closer at zero of
the thirteen states. **The port is right and the mutant is worse.** The
end-to-end table in #1467 says otherwise only because the quantity it measures
puts our bf16 realization and the oracle's bf16 realization on opposite sides of
a shared f32 trajectory; adding a perturbation of the same order as that gap can
land closer by cancellation, and on this fixture it does. That is a property of
the instrument, not of the port. No new issue is filed, because there is no
defect to file.

## Design

1. **`Ltx2PromptConditioning` gains `positions`.** The tower positions the path
   actually ran at, `[num_valid]`, absolute. It is an output of the production
   function and the production function is the only thing that fills it.
2. **The test asserts it against the oracle's own numbering**, `kLtxTowerNumPad + i`
   — a constant the generator emitted from the padded run, not a recomputation
   of the code under test. `first_valid` is separately `REQUIRE`d against the
   same constant three lines above, so neither assertion leans on the other.
3. **The stale note becomes what the table below it measures**, and points at the
   integer assertion as where the coverage now lives.

## Design, continued

4. **What the case enters, and what it does NOT.** It enters at
   `Ltx2EncodePromptToConditioning`, the shipped function, rather than
   constructing a tower by hand, and `out.positions` is not a copy built beside
   the real one: the production function binds
   `std::vector<int32_t>& positions = out.positions` and hands THAT to the tower
   at `ltx2_text_encoder.cpp:1160`, so the assertion reads the bytes that
   executed. Deleting the production write reds it — `M1` below is that
   mutation.

   **This is NOT the reachability mutation `.agents/reachability.md` prescribes,
   and the difference is stated rather than glossed.** MEASURED by the fresh
   review: delete the `ltx2_video.cpp:2278` call site and the focused gate stays
   at 27/27 and 4127/4127, because the case enters one level below
   `Ltx2VideoEngine::Generate`. That entry point is pre-existing — base already
   entered there — and this row adds three assertions to that same case rather
   than a new capability, so it is not what this row owes. It is recorded
   because a reader would otherwise read `M1` as a reachability proof, and it is
   not one. The three call sites are real (`:2278`, `:3007`, `:5510`, in
   `Ltx2VideoEngine::Generate` and `::GenerateAudioOnly`), and what is test-only
   here is the READ of the field, never the write.

## Evidence

One build directory, `/tmp/b1467`, CPU-only Release (`-DVLLM_CPP_CUDA=OFF`,
`-DCMAKE_BUILD_TYPE=Release`, so NDEBUG), Ninja, x86_64, gcc 13, base
`5d638b67e`. Compile rc 0 on every arm. `src/.../ltx2_text_encoder.cpp` pristine
`sha256 8911296882e03ea072a3d6b1898b7fff88503c2f048683802a2a7a43de07a59c`,
restored and re-verified after the mutation.

Rows 1 and 2 are the BASE tree, `5d638b67e`, before this row's assertions
existed; that is why they carry 4118 and not 4127, and why the pristine sha
above is the base file's. Rows 3 and 4 are this row's head.

| # | tree | `test_ltx2_text_encoder` | this case's ratios | reading |
|---|---|---|---|---|
| 1 | base `5d638b67e`, unmutated | `SUCCESS`, 27/27, **4118** of 4118 | video **1.20939x**, audio **1.31288x** | reproduces #1467's `at aeba0de6f / correct` row to three digits |
| 2 | base + `M1`: `positions[i] = i` | `SUCCESS`, 27/27, **4118** of 4118 | video **0.683056x**, audio **0.93133x** | reproduces the `renumbered` row, and **the WHOLE suite is green** — nothing in this file detects it |
| 3 | head + `M1` | `FAILURE`, **8** of 4127 failed | video 0.683056x, audio 0.93133x — still PASSING | RED-first: every one of the T=8 positions fails at `test_ltx2_text_encoder.cpp:2398`, and only that assertion does |
| 4 | head, unmutated | `SUCCESS`, 27/27, **4127** of 4127 | video 1.20939x, audio 1.31288x | green after restore, +9 assertions |

Row 2 is the finding that matters for the note: the trap is not inherited from
#1467, it is measured at this HEAD, and it is worse than #1467 said — not one of
the 27 cases reds.

The oracle re-run behind `## The verdict` is
[`scripts/probe-ltx2-tower-positions.py`](../../scripts/probe-ltx2-tower-positions.py), which imports the committed generator and calls its
own `build_tower` / `run_tower`, so no second implementation of the fixture
exists. Run twice: `transformers 5.12.1` + `torch 2.12.1+cpu` (the goldens'
oracle) and `transformers 5.14.1` + `torch 2.11.0+cu130` with
`CUDA_VISIBLE_DEVICES=""` (the parity pin). Byte-identical results.

## Not done, and named

- **`test_cpu_x86_llamacpp_floor` is FLAKY on this box under load, and that is
  measured rather than assumed.** It failed twice — once in the pre-edit
  `scripts/agent-preflight.sh` on the PRISTINE base with a 12-way build of this
  row saturating the same host, and once in `--staged` with several other agent
  sessions live — always on the same case,
  `CpuX86FloorHarnessTests.test_a_contended_leg_is_discarded_and_never_summarised`.
  Run on its own it passes 10 of 10 in 19.4 s, and the next `--staged` preflight
  reported `All gates green`. This row changes no CPU kernel and no harness. The
  flake is NOT this change; it is also NOT repaired here, and a reader who sees
  it red should re-run before reading it as a verdict.
- **No GPU leg**, because none was authorised for this row and none is needed.
- **`scripts/probe-ltx2-tower-positions.py` is run by no gate**, and it cannot
  be: it needs torch and a `transformers` that registers `gemma4_unified`, which
  CI does not have — the same reason
  `gen-ltx2-gemma-tower-goldens.py` is not run by one either. It reads that
  generator's `build_tower` and `run_tower` signatures, so a change to those
  rots it silently. Recorded because this row's verdict rests on it.
- **`.env` names `VLLM_ORACLE=$HOME/venvs/vllm-oracle` and
  `DEPENDENCY_SOURCE=$HOME/venvs/vllm-oracle/lib/python3.12/site-packages`, and
  neither path exists on this box.** The pin's transformers 5.14.1 is at
  `$HOME/venvs/music3-oracle` instead. Recorded, not repaired: fixing an
  environment value is the developer's to answer, not a value to infer.

## Gates

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build-cpu --target test_ltx2_text_encoder -j 12
./build-cpu/tests/test_ltx2_text_encoder
```

## Stop conditions

Return `NEEDS_DECISION` rather than widening a bound. Return `NEEDS_CONTEXT`
rather than inferring a host, a lease or a preference.
