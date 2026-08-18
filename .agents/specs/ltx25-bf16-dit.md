# LTX-2.5 — the unquantized DiT, and the arms that could not read their own weights

Row: `LTX25-BF16-DIT`. Issue:
[#1148](https://github.com/mudler/vllm.cpp/issues/1148). Owning row:
`MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model`
([model-matrix.md](../model-matrix.md)), phase L6 loaders. Upstream pin:
Lightricks/LTX-2 `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`.

## 0. What is wrong today — measured, not inferred

`PlanDit` (`src/vllm/model_executor/models/ltx2_loader.cpp:414-418` @
`c83b96934`) refuses any DiT checkpoint carrying neither `U8` nor `F8_E4M3`:

```cpp
if (!saw_u8 && !saw_f8) {
  Fail("the DiT checkpoint carries no quantized weights at all (no U8 and no "
       "F8_E4M3). A bf16 DiT is not what phase L6 loads; use the L2 path.");
}
```

**The advice is unreachable.** `Ltx2LoadDitFromSafetensors` *is* the L2 path and
calls `PlanDit` on its first line. `git grep -n 'PlanDit(file)' -- src` returns
three call sites at `c83b96934` — `:653` (`Ltx2ParseDitParamsFromCheckpoint`),
`:660` (`Ltx2LoadDitFromSafetensors`), `:721` (`Ltx2StreamDitToDevice`) — plus
`:788` (`Ltx2RebindDitLoras`), and every one refuses identically. The issue
reports five sites including `:1400`; that coordinate is stale and now points at
`Ltx2LoadVaeWeights`, which is unrelated. Correcting the count here rather than
repeating it.

**The file it refuses.** Read from the header of
`/mnt/nas_share/checkpoints/ltx-2.5/lightricks-ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors`
on 2026-08-17, by parsing the 677,616-byte JSON header and no payload:

| Fact | Value |
|---|---|
| bytes | 42,018,190,584 (data end + 8 + header == file size) |
| tensors | 4349, every one `model.diffusion_model.`-prefixed |
| dtypes | BF16 4059, F32 290 |
| `_scale` / `_scale_2` / `torchao_nvfp4` names | **0** |
| F32 tensors | exactly the six `*scale_shift_table*` families (49+49+48+48+48+48) |
| `__metadata__` | `config`, `gemma_source_checkpoint`, `license`, `model_version` = `2.5.0` |
| layers | 48; `keyframes_abs_pos_embedding` present as BF16 `[1, 4096]` |
| families outside the DiT contract | `video_embeddings_connector` (129), `audio_embeddings_connector` (129) — both already `LoadedElsewhere` |

So it is refused at load, and the 39.13 GiB fetch that unblocked it is unusable.

**What that costs.** Upstream's pipeline table
(`packages/ltx-pipelines/CLAUDE.md:17-30` @ `fd4ded7f`) marks `Full` or
`Full + distilled LoRA` for `TI2VidOneStagePipeline`, `T2AOneStagePipeline`,
`TI2VidTwoStagesPipeline`, `TI2VidTwoStagesHQPipeline`, `A2VidPipelineTwoStage`
and `KeyframeInterpolationPipeline`. This tree has landed `one_stage`,
`t2a_one_stage`, `res2s_two_stage` and `a2vid_two_stage`, and every one of them
could only ever run against a DISTILLED checkpoint — a different sampling regime
that renders plausibly, with nothing validating the class
([#1137](https://github.com/mudler/vllm.cpp/issues/1137)).

## 1. What upstream does, with anchors

Upstream has **no third quantization state**. It has quantized payloads and
everything else, and everything else is the ordinary case.

`packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py:51-58` @
`fd4ded7f`:

```python
# dtypes that ``build(..., dtype=)`` may cast. Quantized payloads (uint8 NVFP4,
# float8, etc.) must not be rewritten — SDOps policies emit them at load time.
_DTYPE_CASTABLE = frozenset(
    {torch.float32, torch.float64, torch.float16, torch.bfloat16}
)
```

`_load_model_weights` (`:74-95`) then calls `meta_model.load_state_dict(sd,
strict=False, assign=True)` on whatever the file holds. There is no branch on
"is this quantized"; the quantized arms are what the SDOps policies emit, and
bf16 is what a checkpoint carries when no policy ran.

`README.md:73` @ `fd4ded7f` names
`ltx-2.5-22b-dev-transformer-bf16.safetensors` as "the full model; used by the
guided two-stage pipelines".

**So the mirror is:** `Ltx2DitQuant` grows `kNone`, meaning *not quantized*, and
it is the baseline the other two are exceptions to — not a new scheme.

## 2. What is actually behind the refusal

Nothing past `PlanDit` had ever been exercised on a pure-bf16 file, so this was
read rather than assumed before the scope was fixed.

| Callee | Does it need bf16 work? | Why |
|---|---|---|
| `ParseLtx2DitParamsFromManifest` (`ltx2.cpp:364-439`) | no | reads shapes only; the U8 width-doubling happens in `PlanDit`, before it, and a BF16 tensor is never doubled |
| `MaterializeDitTensor` (`ltx2_loader.cpp:424-500`) | **no — the branch already exists** | `:454-462` handles `BF16` with a byte-count check and a `memcpy`, and has been live since L6 for every bias and q/k norm |
| `IsScaleSidecar` (`:360-363`) | no | filters nothing in a file with no sidecar; no contract name ends in `_scale` |
| `UnportedFamilies` (`:573-586`) | no | the dev file's only non-contract families are the two connectors, which `LoadedElsewhere` already excludes |
| `FuseLorasInto` / `Ltx2RebindDitLoras` (`:529-539`, `:763`) | no | keyed on the dtype `MaterializeDitTensor` returned, which is `kBF16` on all three arms |
| `Ltx2WidenDitToF32` (`:694-710`) | no | keyed on `kBF16` views |
| `Ltx2StreamDitToDevice` (`:712-761`) | no | uploads whatever `MaterializeDitTensor` produced |
| `Ltx2DitForwardDevice` (`ltx2_device.cpp:1111-1118`) | no | already accepts `kBF16` or `kF32` |
| `Ltx2DitCheckpoint::quant` consumers | none in `src/` | the field is reported, never branched on |

**The scope therefore held.** The wall is one `if`, and the work is making the
decision have three outcomes instead of two while keeping the mixed-file refusal
that shares the block. This is recorded because the dispatch expected it to be
larger and asked for `NEEDS_DECISION` if it were; it is not.

## 3. Design

`PlanDit` collects the **dtype set** of the non-sidecar tensors as it walks the
header, and resolves:

| File carries | Outcome |
|---|---|
| U8 and F8_E4M3 | refuse — unchanged, the two arms use different sidecars |
| U8 | `kNvfp4` — unchanged |
| F8_E4M3 | `kFp8` — unchanged |
| neither, and at least one dtype this loader reads (`BF16`, `F32`) | **`kNone`** |
| neither, and nothing this loader reads | refuse, **naming the dtypes the file holds and the ones this loader reads** |

The last row is the corrected refusal. It is reachable — a float16 DiT is legal
upstream (`_DTYPE_CASTABLE` lists `torch.float16`) and this port has no F16
materialization — and it never sends the reader to a path that refuses
identically.

`Ltx2DitQuant::kNone` is added **first** in the enum with the upstream anchor
beside it. `Ltx2DitCheckpoint::quant` keeps its `kFp8` default member
initializer: it is assigned on every path, and moving it would be an unrelated
behaviour change.

**No new materialization code.** The bf16 arm reaches `MaterializeDitTensor`'s
existing `BF16` branch, which is the same branch every bias on the FP8 arm
already takes. That is what keeps this from being a second loader.

## 4. Memory format — check this explicitly

Per [`porting.md`](../porting.md) §*Mirror the memory format*:

| Ask | Answer for this arm |
|---|---|
| what dtype does the load OUTPUT? | `kBF16` for every weight, `kF32` for the six `scale_shift_table` families — identical to the FP8 and NVFP4 arms, because those dequantize to bf16 |
| does anything widen? | only `Ltx2WidenDitToF32`, opt-in, for `Ltx2DitForward`'s f32 parity arm; unchanged |
| does the file's own dtype survive? | yes — `BF16` in, `kBF16` out, `memcpy`, no round trip. The F32 tables stay F32 because the CHECKPOINT stores them F32 |
| bytes per model | 42,018,190,584 host, against ~18.7 GB on disk for the distilled NVFP4 copy which materializes to ~21 GB bf16. The dev file is bigger because it is 21.004 B parameters at 2 bytes, not because anything widened |

The gate is bit-exact against bf16 the fixture wrote, so a widened
materialization fails on the `REQUIRE(t.dtype == vt::DType::kBF16)` rather than
passing a value comparison — which is the one thing a value gate cannot see.

## 5. Tests

Red-first. Every case runs as part of a WHOLE binary; no `--test-case` filter.

`tests/vllm/models/test_ltx2_loader.cpp`:

1. **the value gate** — `BuildSyntheticDit(p, kNone, {})` writes every rank-2
   non-table weight BF16 with `TrueValue`-derived content and no sidecar. The
   load is compared **bit-for-bit** over every contract weight, and the
   expectation itself is checked for the two shapes a stub hits by accident:
   `want_nonzero == want_total` (a zero fill scores 0) and `distinct > 1000` (a
   constant fill scores 1). `rank2 == unquantized_weight_tensors` proves the
   comparison actually covered the tensors the quantized arms take a different
   branch for. The F32 tables are checked separately at `max_abs == 0`.
2. **the mixed refusal survives** — an FP8 file with one module swapped for the
   NVFP4 arm's U8 + two sidecars is still refused. This is the branch that must
   NOT become `kNone`, gated beside the one that must go. It was green before
   this row and is a regression guard.
3. **the corrected refusal** — every `BF16` entry retyped to `F16` (same width,
   so no byte count changes). The message must contain `F16` and must NOT
   contain `L2 path`.
4. **the LoRA hook serves the third arm** — `CheckArmFuses(kNone, "bf16")`,
   the same element-wise delta claim the FP8 and NVFP4 arms already make.

`tests/vllm/multimodal/test_ltx2_video.cpp`:

5. **reachability** — `ReducedDitOptions::unquantized` writes the workspace DiT
   with no quantized tensor and no sidecar (asserted from the file before the
   engine sees it, so a fixture flag that did nothing cannot pass), then
   `LoadVideoEngine` on its DEFAULT configuration loads it and renders phase 0.
   Frames at the claimed size carrying more than one byte value, and a waveform
   that is not digital silence — the same floor the FP8 render case uses,
   because an all-NaN decode serializes as a well-formed black frame.
6. **real weights, header only** — a subcase of the `LTX2_CHECKPOINT_ROOT`-gated
   case asserts the shipped dev file resolves to `kNone`, recovers 48 layers /
   4096 / 2048 / 128 / 128, carries `keyframes_abs_pos_embedding` TRAINED, holds
   0 U8, 0 F8_E4M3, 0 sidecars, 4059 BF16 and 290 F32, and that its declared
   config adopts onto the identical weight contract.

## 6. Risks

- **A half-quantized file resolving to `kNone`.** It cannot: `saw_u8` and
  `saw_f8` are OR-reductions over the whole header, so one quantized tensor is
  enough to select an arm, and both set means refuse. Test 2 holds it.
- **A dtype this loader cannot read loading as zeros.** It cannot:
  `MaterializeDitTensor` refuses an unknown dtype BY NAME, and `PlanDit` now
  refuses earlier when nothing in the file is readable. Test 3 holds it.
- **The bf16 arm silently widening.** Held by `REQUIRE(t.dtype ==
  vt::DType::kBF16)` in test 1, which a value comparison alone cannot see.
- **The fixture flag doing nothing.** Held by counting the file's own dtypes
  before the engine opens it, in tests 1 and 5.

## Owed

- **A real-weights MATERIALIZATION of the dev transformer.** Test 6 parses the
  header and stops there, because `Ltx2LoadDitFromSafetensors` on this file
  materializes ~42 GB of host bf16 and the CPU gate cannot hold it. What is
  owed is a load and a render on the full model, on a box that can. `dgx:gpu0`
  is `unhealthy` and `thor:gpu0` / `orin:gpu0` cannot hold 42 GB, so no fleet
  device closes it today. Tracked by
  [#1048](https://github.com/mudler/vllm.cpp/issues/1048), which owns the
  LTX-2.5 checkpoint pin and the "no LTX-2.5 arm has been rendered on real
  weights" statement in `docs/USAGE.md`.
- **The checkpoint CLASS is still unvalidated** —
  [#1137](https://github.com/mudler/vllm.cpp/issues/1137). This row makes the
  full model READABLE; it does not make an arm refuse a distilled checkpoint it
  was not written for. A `res2s_two_stage` load against a distilled file still
  renders in the wrong sampling regime with no diagnostic. Out of scope here
  deliberately: it needs a class signal the header does not obviously carry, and
  guessing one is how a correct checkpoint gets refused.
- **The memory envelope is unmeasured.** 42 GB bf16 against the distilled copy's
  ~21 GB is arithmetic, not a measurement, and nothing has run either through
  `Ltx2StreamDitToDevice` on a device that could hold them.
- **Four of the five DiT artifacts cannot be pinned by CONTENT from here** —
  [#1048](https://github.com/mudler/vllm.cpp/issues/1048). See §8; it needs an
  authenticated hub fetch.

## 8. The hub's content hash for this repo is a FABRICATION, and it type-checks

Found while writing the `docs/USAGE.md` pin AGENTS.md §*Say which weights, and
from where* requires. `Lightricks/LTX-2.5` is a **gated** repo: an
unauthenticated `resolve` returns `Access to model Lightricks/LTX-2.5 is
restricted`. Its `/api/models/.../tree` listing still answers, still carries
real `size` values — and returns an `lfs.oid` that is **one character repeated
64 times**.

It is 64 characters. It is lowercase hex. `re.fullmatch(r"[0-9a-f]{64}", oid)`
passes. A pinning script that reads that field writes five different checkpoints
into a document under one invented digest and exits 0.

The tell is cheap and it is the only one: **all 14 LFS files in the repo return
the SAME oid**. The first draft of this row's `docs/USAGE.md` table carried five
sha256 columns filled from that field, and the assertion that caught it was
`dev_oid != distilled_oid` — written for an unrelated reason, because the two
bf16 transformers are the same SIZE (42,018,190,584 bytes each) and the row
wanted to say that a size check cannot separate them. That assertion firing is
what turned a plausible table into a measured one.

So `docs/USAGE.md` publishes exactly one sha256 — the LOCAL copy's,
`792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584`, over all
42,018,190,584 bytes — labelled as the local copy, because there is nothing
here to compare it against. The other four rows say "not obtainable here"
rather than carrying a number.

This is the [#1148 lesson](https://github.com/mudler/vllm.cpp/issues/1148) in a
second place: an instrument that fails toward a well-formed answer is worse than
one that errors, and `len(oid) == 64` is not a check.

## 7. Stop conditions

1. The scope is larger than one `if` plus its tests — return `NEEDS_DECISION`
   rather than landing a path that loads and then materializes wrong numbers.
   **Resolved: it is not.** §2 is the evidence.
2. Any existing arm's goldens move. Nothing here touches the FP8 or NVFP4
   branches, and the full gate is the check.
3. The corrected refusal cannot be made truthful without inventing a class
   signal. It can: it names the dtypes the file holds.

## 9. Mutations

The harness ran inside the build tree and is not committed, so its method is
recorded here rather than by a path that no longer exists. For each row it
substitutes one anchor, then prints four facts, because three of them have
separately produced a green-that-proves-nothing in this campaign:

- `git diff --stat` for the mutated file — a mutation that never applied reads
  as a passing test;
- whether the target BUILT, and the `: error:` count — a mutation that fails to
  build reads as a passing test;
- the exit code captured DIRECTLY from the process, never through a pipe;
- doctest's `test cases:` and `Status:` lines, since a thrown case prints
  `0 failed` beside `Status: FAILURE!`.

The anchor is asserted to occur EXACTLY ONCE in the file before substitution: a
non-unique anchor mutates a different function on a sibling row and presents as
a compile error about the code under test. After each row the original text is
written back, the file's mtime is bumped so ninja cannot skip the rebuild, and
the restored file's sha256 is compared against the pre-mutation tree. Whole
binaries are run; no `--test-case` filter is used anywhere, because a filter
matching zero cases prints `SUCCESS!` at exit 0.

| # | Mutation | Target | Built | `: error:` | Exit | Result |
|---|---|---|---|---:|---:|---|
| M1 | restore the refusal this row removed | `test_ltx2_loader` | yes | 0 | 1 | DETECTED (3 of 37 cases) |
| M1 | " | `test_ltx2_video` | yes | 0 | 1 | **DETECTED (1 of 80)** — the reachability proof |
| M2 | materialize bf16 weights as ZEROS | `test_ltx2_loader` | yes | 0 | 1 | DETECTED (5 of 37) |
| M3 | widen the bf16 arm to f32, values correct | `test_ltx2_loader` | yes | 0 | 1 | DETECTED (5 of 37) |
| M4 | let a mixed U8+F8 checkpoint through | `test_ltx2_loader` | yes | 0 | 1 | DETECTED (1 of 37) |
| M5 | put "use the L2 path" back in the refusal | `test_ltx2_loader` | yes | 0 | 1 | DETECTED (1 of 37) — **after a repair, see below** |
| M8 | the dtype refusal never fires | `test_ltx2_loader` | yes | 0 | 1 | DETECTED (1 of 37) |
| M6 | make the fixture's `unquantized` flag inert | `test_ltx2_video` | yes | 0 | 1 | DETECTED (1 of 80) |
| M7 | delete the production DiT call site | `test_ltx2_video` | yes | 0 | 1 | DETECTED (61 of 80) |

**M5 CHANGED A TEST, WHICH IS WHY IT WAS RUN.** On the first pass it was NOT
DETECTED: 37 of 37 passed at exit 0 with "use the L2 path" back in the surviving
refusal. The cause was in the test, not the code. The F16 case retyped only the
`BF16` entries, which left the F32 `scale_shift_table` families in place — so
`PlanDit` still saw a dtype it reads, resolved `kNone`, and the refusal that
fired came from `MaterializeDitTensor` instead. Both messages satisfy both of
that case's assertions, so it passed while the branch it exists for was never
reached, and the new `PlanDit` refusal was unexercised code with a message
nothing checked.

The repair retypes EVERY entry including the tables, halving the F32 payloads
because `safetensors_reader.cpp:165-173` cross-checks `numel * dtype_size ==
nbytes`, and asserts the message is `PlanDit`'s (`no weight this loader can
read`) before asserting anything about its content. M5 and M8 both DETECT after
it. This is the third shape of "green proves nothing" this campaign has hit and
the first where the passing test named the right symptom for the wrong reason.

## Now

`ACTIVE`. Spec committed before implementation; the implementation commit
follows it on `row/LTX25-BF16-DIT`.
