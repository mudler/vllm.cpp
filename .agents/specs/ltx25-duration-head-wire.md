# LTX25-DURATION-HEAD-WIRE — the duration head gets a driver

Row: `LTX25-DURATION-HEAD-WIRE`. Issue: #2900.
Upstream pin: `Lightricks/LTX-2 @ fd4ded7f` (`.agents/oracles/ltx-2.md`).

## Now

`ACTIVE`, fresh review returned PASS WITH REPAIRS and the repairs are applied.
The gap this row opened against: the head's math had been ported and gated since
phase L5 with nothing constructing one, so `duration_head_path` was refused by
name and an auto duration silently became the recipe default. The driver landed,
and `CheckUnservedExtras` no longer names the key.

The repairs are four. The `None`-predictor polarity is now measured through its
CONSEQUENCE rather than through a load that succeeds either way (§5 T2, and
mutation M5 in `## Outcome` is what said the original cases measured nothing).
Three invented refusals around `MIN <= 0` are gone (§3.4). An `auto_duration` on
a retake refuses by name instead of being dropped (§3.4). And the product prose
this row falsified is repaired in the same change, including the user-facing
`Fail(...)` in `ltx2_t2a.cpp` that sent a caller to `num_frames` when the answer
is `duration_head_path`.

## Scope

**In.** `DurationPredictor` and the two pipeline-level functions that surround
it, mirrored from `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py`:
construction from a checkpoint including its `None` case, the seconds-to-frames
conversion with its clamp and its `8k + 1` snap, the top-of-call raise, and the
resolution point after prompt encoding. Then the wiring that makes
`Ltx2VideoEngine::Generate` reach all of it.

**Out.** The head's **bf16 arm**. Upstream constructs the head in the pipeline
dtype (`distilled.py:163-167` passes `self.dtype`, which `:109` resolves to
`torch.bfloat16`); this port materializes f32, as `ltx2_duration_head.h` already
records. That arm is the eighth component of gap A24
(`.agents/specs/ltx25-completion-scope.md` §A.7) and it is listed under `## Owed`
below. It is deliberately not this row: **a dtype arm cannot be gated until
something reaches it**, and supplying that reach is this row's whole subject.

**Out.** Any change to `Ltx2DurationPredict`, `Ltx2DurationAttentionPool` or
their goldens. The head's arithmetic is already gated and this row does not
touch it.

## 1. The gap, stated as the two things that are different

`include/vllm/model_executor/models/ltx2_duration_head.h` is a complete port of
`DurationHead.forward` (`duration_head.py:89-118`). It works. Nothing calls it
from a production path.

Two sites say so in the tree today:

- `src/vllm/multimodal/ltx2_video.cpp:605` `CheckUnservedExtras` refuses
  `duration_head_path` **by name**, because `kKnownLoadExtras` accepts the key
  and no code reads it.
- `src/vllm/multimodal/ltx2_video.cpp:3162-3177` resolves a frame count by exact
  arithmetic against the recipe frame rate, with a comment naming the auto path
  as the missing one.

This is the `## Nothing lands dead` shape in its second form: not a seam that
cannot represent the behaviour, but a capability that no production entry point
reaches.

## 2. Upstream anchors, each read at `fd4ded7f`

`packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py`:

| Upstream | Line | What it decides |
|---|---|---|
| `class DurationPredictor` | `:803` | holds a **built** head; `__init__` (`:813`) takes one directly |
| `DurationPredictor.from_checkpoint` | `:816` | returns **`None`**, never raises, when the path is `None` or the file has no head |
| `DurationPredictor.__call__` | `:850` | seconds -> clamp -> `8k + 1` snap; refuses a non-single-item batch |
| `require_num_frames_source` | `:894` | raises at the **top** of `__call__`, before any work |
| `resolve_num_frames` | `:908` | resolves **after** prompt encoding, from the connector outputs |

`packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`:

| Upstream | Line | What it decides |
|---|---|---|
| `snap_frames_to_grid` | `:554-562` | floors to `k * time_scale + 1`; refuses `frames < 1` |
| `seconds_to_clamped_num_frames` | `:565-585` | rounds, clamps **before** snapping, then snaps **up** if the floor undershot `min_frames` |

`packages/ltx-pipelines/src/ltx_pipelines/utils/types.py:116` `AutoDuration`,
defaults `min_seconds=1.0`, `max_seconds=20.0`; `DEFAULT_AUTO_DURATION` at `:126`.
`packages/ltx-pipelines/src/ltx_pipelines/utils/args.py:108` `AutoDurationAction`
parses `--auto-duration MIN_SECONDS MAX_SECONDS` and refuses `MIN > MAX`.
`packages/ltx-core/src/ltx_core/duration_head/model_configurator.py:9-11`
`DURATION_HEAD_KEY_OPS` matches the prefix `duration_head.` and strips it.

Six pipelines construct a predictor and call both functions:
`ti2vid_one_stage.py:126,154`, `ti2vid_two_stages.py`, `ti2vid_two_stages_hq.py:144,197`,
`distilled.py:163,211,231`, `dfr_pipeline.py:262,289,306`, and
`t2a_one_stage.py:103,123` — the last passing `video_encoding=None`, which is why
`DurationHead.forward` takes either stream.

**Two details worth stating because they are easy to get backwards.**
First, `num_frames` **defaults to `DEFAULT_AUTO_DURATION`** upstream
(`distilled.py:195`, `dfr_pipeline.py:276`), so upstream's omission means
auto-predict, not a default count. Second, the two-stage pipelines return the
resolved count as the **third of four** tuple elements
(`ti2vid_two_stages.py:181,312`), and their CLI feeds it to
`get_video_chunks_number` at `:374`; the one-stage pipeline returns a **3-tuple
without it** (`:245`) and its CLI falls back to `args.num_frames` at `:305`. The
asymmetry is upstream's and this row does not import it — `VideoResult` already
carries the rendered geometry.

## 3. Design

### 3.1 `Ltx2LoadDurationHeadWeights` — upstream's `None`, not an error

New in `ltx2_duration_head.{h,cpp}`, shaped on `Ltx2LoadConnectorWeights`
(`ltx2_loader.cpp:1534`) because it solves the same problem against the same
`Ltx2VaeWeights` bag. **This section first said `ltx2_loader.{h,cpp}` and the
loader is not where it went**: the head's file is a plain safetensors bag with no
quantization plan and no shards, so it reads directly rather than through the
DiT planner, and it sits beside the forward it feeds:

```
bool Ltx2LoadDurationHeadWeights(const SafetensorsFile& file,
                                 const Ltx2DurationHeadConfig& config,
                                 vt::DType compute_dtype, Ltx2VaeWeights* out);
```

`false` is upstream's `None`. Upstream builds with `strict=False` and then asks
`any(param.is_meta for param in head.parameters())` (`blocks.py:838`), so a
checkpoint missing **any** head tensor yields `None` rather than an exception —
a partial head is `None` too, and that polarity is mirrored exactly. A tensor
that is **present with the wrong shape** is a different thing and refuses by
name, as the connector loader does: upstream's `Builder` would not silently bind
it either.

`compute_dtype` is `kF32` here and the parameter exists so the bf16 arm is a
call-site change rather than a signature change. It refuses anything but `kF32`
and `kBF16` by name.

### 3.2 The frames arithmetic

New in `ltx2_duration_head.{h,cpp}`, beside the forward it consumes:

```
int64_t Ltx2SnapFramesToGrid(int64_t frames, int64_t time_scale);
int64_t Ltx2SecondsToClampedNumFrames(double seconds, double frame_rate,
                                      int64_t min_frames, int64_t max_frames,
                                      int64_t time_scale);
int64_t Ltx2DurationPredictFrames(const Ltx2DurationHeadConfig& config,
                                  const Ltx2VaeWeights& weights,
                                  const float* video_tokens, int64_t video_token_count,
                                  const float* audio_tokens, int64_t audio_token_count,
                                  double frame_rate, double min_seconds, double max_seconds,
                                  int64_t time_scale, float* predicted_seconds);
```

The last two parameters were not in this section's first draft and both are
load-bearing. `time_scale` is the VAE's causal temporal factor, passed in rather
than assumed so the auto-duration snap and the latent shapes cannot disagree
about it. `predicted_seconds` returns the RAW prediction beside the frame count,
which is what lets a mismatch say whether the forward or the snapping moved
instead of arriving as one wrong integer; `Ltx2ConditioningTrace` carries it, and
T7 asserts against it.

**Three rules here fail silently and each is gated separately** (§5):

1. **The clamp precedes the snap** (`helpers.py:579-580`). Snapping first and
   clamping after gives a different count for every prediction whose floored
   grid point falls outside the window, and both orders type-check.
2. **The undershoot repair snaps UP, and is itself capped by `max_frames`**
   (`:581-584`). `snap_frames_to_grid` floors, so a `min_frames` that is not
   already a grid point produces a count **below** the contract; upstream
   recovers with a ceiling division back onto the grid, then takes `min` with
   `max_frames`. Dropping that `min` breaks the window in the other direction on
   a degenerate range.
3. **`round` AS PYTHON DEFINES IT — half-to-EVEN — and not truncation**
   (`:578`). Two rules are wrong here, not one. `(int)(seconds * frame_rate)`
   truncates and disagrees with `round` on roughly half of all inputs; and
   `std::llround`, which is the obvious repair, rounds a half AWAY FROM ZERO
   where Python takes the even neighbour. 0.34 s at 25 fps is exactly 8.5:
   upstream takes 8 and snaps to frame 1, `llround` takes 9 and snaps to frame 9.
   Eight frames, on a request a user can type. The goldens carry that case and
   0.5 s at 49 fps beside it, with `llround`'s answer in a
   `rejected_half_away` column.

The batch refusal (`blocks.py:857-861`) is mirrored: this port takes `batch` out
of the signature entirely and always predicts one item, which is the same
contract expressed so it cannot be violated.

### 3.3 `Ltx2RequireNumFramesSource` — the position IS the behaviour

```
void Ltx2RequireNumFramesSource(bool auto_requested, bool has_predictor);
```

Called from the **top** of `Ltx2VideoEngine::Generate`, in the `generate.setup`
phase, before prompt encoding and before any weight is touched. Upstream is
explicit that this is the point (`blocks.py:896-899`: "Call at the very top of a
pipeline's `__call__` — before prompt encoding or any other work — so a
checkpoint without DurationHead weights fails fast"). A refusal that arrives
after the encode is a different behaviour from upstream's even though it refuses
the same requests, and §5's T6 gates the position, not only the message.

### 3.4 The request surface

Upstream's user-facing spelling is `--auto-duration MIN_SECONDS MAX_SECONDS`
(`args.py:108`). Mirrored as a **per-generation extra** `auto_duration`, valued
`"MIN,MAX"`, which needs no ABI change — the parallel-array shape
`vllm_video_params` already carries is what that shape exists for. `MIN > MAX`
refuses by name, as `AutoDurationAction` does.

**Omitting every frame source auto-predicts when, and only when, a head is
loaded.** That is upstream's polarity (`num_frames` defaults to
`DEFAULT_AUTO_DURATION`) reached without changing any existing caller: an engine
loaded with no `duration_head_path` keeps today's recipe default exactly, and the
behaviour changes only for a caller who opted in by naming a head. Stating the
alternative so the choice is visible: making omission mean AUTO unconditionally
is the closer mirror of the signature and it would turn every existing
LTX-2.5 render on a pre-2.5 checkpoint into a `require_num_frames_source`
refusal. This port has a recipe where upstream has a required argument, and the
recipe is what stands in for the caller's explicit count.

`auto_duration` and an explicit `num_frames`/`duration` together refuse by name
rather than one silently winning. **So does `auto_duration` with a retake
window**, and the alternative is why: `retake.py` takes its length from the
source clip's metadata (`:220`) and constructs no `DurationPredictor` at all, so
there is nothing upstream to mirror and the extra would simply be dropped — the
same silent win the explicit-count case refuses. The IMPLICIT auto request — no
count, a head loaded — is NOT refused there, because it asked for nothing and the
retake's own geometry is what an omitted count resolves to. The asymmetry is
recorded here rather than left for a reader to find.

**ONE REFUSAL THIS SECTION FIRST INVENTED, now removed.** `ParseAutoDuration`
refused `MIN <= 0`. `AutoDurationAction` (`args.py:117-122`) refuses `MIN > MAX`
and nothing else, and at the pin `AutoDuration(min_seconds=0.0,
max_seconds=20.0)` constructs while `seconds_to_clamped_num_frames(3.0,
frame_rate=25.0, min_frames=0, max_frames=500)` returns 73 — so
`--auto-duration 0 20` is a working upstream request this port refused by name.
The same polarity appeared twice more: `Ltx2SecondsToClampedNumFrames` required
`min_frames >= 1`, and `Ltx2DurationPredictFrames` floored the rounded bound to
1. Upstream does neither. What it does instead is RAISE where the clamped count
reaches zero — `seconds_to_clamped_num_frames(0.005, frame_rate=25.0,
min_frames=0, max_frames=500)` is `ValueError: frames must be >= 1, got 0`, out
of `snap_frames_to_grid` rather than out of a bound check — and the floor turned
that raise into a silent one-frame render. All three are gone, and
`test_ltx2_pipeline` gates both directions: the accepted bound AND the raise,
because accepting the bound without honouring the raise is the silent half.

### 3.5 The two resolution sites

Both are `resolve_num_frames`, at upstream's position — **after** prompt
encoding, where the connector outputs exist:

- `ltx2_video.cpp:3162` — the video path, both streams.
- `ltx2_video.cpp:6065` — `GenerateAudioOnly`, which mirrors
  `T2AOneStagePipeline` and passes **`video_encoding=None`**
  (`t2a_one_stage.py:103`). This is why the port keeps a null video stream
  reachable rather than requiring both.

## 4. Risks

- **The prefix.** `DURATION_HEAD_KEY_OPS` strips `duration_head.`, so a file
  storing bare names and a file storing prefixed names are both real. The config
  already carries a `prefix` field; the loader tries the prefixed spelling and
  falls back, and T2 gates both.
- **A silently-wrong window.** A clamp that never fires reads exactly like a
  clamp that is correct. T4 probes **inside and outside** the window and asserts
  the rejected rule's answer differs.
- **Reachability regression.** The capability is reachable today only through the
  new call sites; T6 and T7 delete them in a scratch copy and must red.
- **Concurrent-edit false red.** `test_ltx2_video`'s reader-anchor case reads
  `ltx2_video.cpp` at RUN TIME, so a mutation left applied reds it for an
  unrelated reason. Every mutation is restored **and rebuilt** before the next
  run.

## 5. Tests

Goldens come from `scripts/gen-ltx2-duration-wire-goldens.py`, which **executes**
the pinned upstream module on synthetic weights — it does not transcribe it.
Every case emits upstream's answer **and the nearest rejected hypothesis**, and
refuses to write when the two are not separated.

| ID | Claim | Shape |
|---|---|---|
| T1 | `Ltx2SnapFramesToGrid` floors to `8k + 1` and refuses `frames < 1` | value golden + refusal |
| T2 | the loader returns upstream's `None` for an absent **and** a partial head, and refuses a shape mismatch by name | synthetic safetensors, four files: the headless DiT, a whole head, a partial head, a wrong-shaped head |
| T3 | `Ltx2SecondsToClampedNumFrames` clamps **before** snapping | golden + the snap-first answer, `separating > 0` |
| T4 | the undershoot repair snaps **up** and is capped by `max_frames` | golden + the un-repaired answer |
| T5 | `Ltx2DurationPredictFrames` reproduces upstream's frame count end to end | golden over the executed chain |
| T6 | the auto request refuses at the **top** of `Generate`, before any work | ordering assertion, not only the message |
| T7 | `duration_head_path` is **served**: a head named at load is opened and reaches the frame count | production path, `Load` + `Generate` |
| T8 | `auto_duration` refuses `MIN > MAX`, and refuses combining with an explicit count | refusal by name |

**Every claimed guarantee is mutated before it is claimed** — five of five A24
waves shipped one that mutation showed was not measured. Each mutation is
confirmed **applied** and **compiled** before its result is read, because a
mutation that does not build reads as a passing test.

## 6. Gates

Run from the row worktree.

```sh
# the upstream oracle re-executes and the goldens are unchanged
python3 scripts/gen-ltx2-duration-wire-goldens.py --ltx2 /home/mudler/_git/LTX-2 \
    --out tests/vllm/models/ltx2_duration_wire_goldens.inc
git diff --exit-code tests/vllm/models/ltx2_duration_wire_goldens.inc

# determinism: the generator is thread-count independent (#2855)
OMP_NUM_THREADS=1 python3 scripts/gen-ltx2-duration-wire-goldens.py \
    --ltx2 /home/mudler/_git/LTX-2 --out /tmp/dw1.inc
OMP_NUM_THREADS=8 python3 scripts/gen-ltx2-duration-wire-goldens.py \
    --ltx2 /home/mudler/_git/LTX-2 --out /tmp/dw8.inc
cmp /tmp/dw1.inc /tmp/dw8.inc

# the focused suites
ctest --test-dir build -R 'ltx2_video|ltx2_pipeline' --output-on-failure

# the full gate, and the size check preflight skips
scripts/agent-preflight.sh
python3 scripts/check-pr-size.py --base origin/main --head HEAD
```

## 7. Evidence

Recorded in `## Outcome` at `DONE`: the literal red before each change, the
literal green after, every mutation with its literal result including the
production-call-site deletion, and the gate log lines rather than an exit code.

## 8. Stop conditions

- Return `NEEDS_DECISION` if the auto-duration request surface turns out to need
  an ABI change rather than an extra.
- Return `NEEDS_DECISION` rather than widening a golden tolerance to pass.
- Do **not** implement the bf16 arm here, and do not gate it by proxy.

## Owed

- **The duration head's bf16 arm.** Upstream constructs the head at the pipeline
  dtype, `torch.bfloat16` (`distilled.py:109,163-167`); this port materializes
  f32. Eighth component of gap A24,
  `.agents/specs/ltx25-completion-scope.md` §A.7. Tracked by #2900 until A24's
  wave for it opens a row. `Ltx2LoadDurationHeadWeights` already takes the
  `compute_dtype` the arm needs, so the arm is a call-site change.
  **This listing IS the ownership record.** #2900 closes when this row lands, so
  it cannot also be the tracker; `.agents/specs/ltx25-a24-leaves-bf16.md` and
  `.agents/specs/ltx25-a24-upsampler-bf16.md` name the same arm under their own
  `## Owed`, and this entry is what a reader of A24's wave chain arrives at.
- **`Ltx2VaeWeights::bf16` is unpopulated for this component**, for the same
  reason.
- **Upstream's four-tuple return** (`ti2vid_two_stages.py:312`) is not mirrored;
  `VideoResult` carries the rendered geometry instead. Named here so the
  difference is recorded rather than discovered.
