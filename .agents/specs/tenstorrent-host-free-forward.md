# Tenstorrent host-free decode forward — plan

Status: **DRAFT plan, 2026-08-13.** The prerequisite for decode mesh-trace
capture (see `tenstorrent-trace-runner.md`: capture aborts on `to_vector`
readbacks inside `ForwardLayers`). This document decomposes the work into
independent rows sized for parallel claims.

Row id: `BACKEND-TENSTORRENT-HOST-FREE-FORWARD` (child of
`BACKEND-TENSTORRENT`). Issue:
[#1105](https://github.com/mudler/vllm.cpp/issues/1105).

## Goal

Make the per-decode-layer region of the TT forward **host-free**: zero
`to_vector` / `EnsureHost` readbacks between `BeginCapture` and
`EndCaptureGraph`. Only then can `Qwen3DenseDecodeGraph` capture/replay it
on `kTENSTORRENT` (ttnn `begin_trace_capture` prohibits any host read).

The per-layer op sequence (Qwen3-dense / Mistral, from
`dense_attn_block.h`) and its host-readback status at T=1 today:

| per-layer op | status today (T=1) | in captured region? |
|---|---|---|
| `RmsNorm` (pre-attn q-norm + residual merges) | HOST (rows<32) | yes |
| `MatmulBT` (qkv / o_proj / mlp) | device | fine |
| `QkvSplit` | **pure host** | yes |
| `RmsNorm` (qk-norm, Qwen3 only) | HOST | yes |
| `RopeNeox` / `RopeFromCache` | HOST (T·H<64) + `BuildCosSinFromPositions` host | yes |
| `ReshapeAndCache` | **pure host** | yes |
| `PagedAttention` | **pure host (host oracle)** | yes |
| `SiluAndMul` | device | fine |

Boundary ops OUTSIDE the layer loop (readbacks here are fine — they are the
capture region's input/output edges): `Embedding` (host-staged upload),
`GreedyArgmax` (host readback of the final logits).

## Scope

**In.** Make the per-decode-layer region of the TT forward host-free (zero
`to_vector` readbacks and zero `enqueue_write` between `BeginCapture` and
`EndCaptureGraph`) so ttnn mesh-trace can capture/replay decode. R1-R3b
landed; item 5 (persistent device input tensors + before-replay populate)
open.

**Out.** Prefill capture, MoE, new ttnn kernels, upstream tt-metal changes
(the answer is a vllm.cpp-side architecture port).

## Upstream chain

No upstream vLLM equivalent. The loyal anchors are: ttnn trace
(`ttnn::operations::trace::{begin,end}_trace_capture`, wired in
`tenstorrent_backend.cpp:70-76`), the CUDA decode-graph capture contract
(`cuda_backend.cu:184-197`: async region, no host sync, no malloc, fixed
pointers), and the tt-metal vLLM plugin's trace design (the reference
implementation of trace-based decode on this hardware).

## Our baseline

Landed on this branch (measured on real Blackhole P150, env-gated
`VT_TT_HOST_FREE_DECODE`, inert by default — 21/21 TT tests): R1 threshold
flip, R2 device->device copy, R3 program-cache warm, R3b device zero-fill.
Capture enters the forward and reaches the layer ops. The open gap is item
5: per-op `enqueue_write` during capture; the fix (persistent device
tensors + before-replay populate) is the plugin-port above. Full measured
record: `tenstorrent-host-free-r1.md`.

## Port map

No upstream vLLM equivalent (no vLLM Tenstorrent platform). The architecture
is ported from the official Tenstorrent vLLM plugin
(`tt/vllm/plugins/vllm-tt-plugin/.../model_runner.py`):

| plugin technique | vllm.cpp TT mapping |
|---|---|
| two-phase warmup (compile ops with `enable_trace=False`, then capture) | `Qwen3DenseDecodeGraph` eager step then capture (already landed) + `device.enable_program_cache()` (R3, landed) |
| persistent device tensors at warmup max-padded shape (stable addresses) | TT decode-graph `SizeSlot` holds persistent ttnn device tensors for inputs (open — item 5) |
| `copy_host_to_device_tensor` before capture/replay, never inside | populate the stable buffers via `ttnn::copy_to_device` before `ReplayGraph` (open — item 5); inside the captured region only `CopyDeviceDeviceIfCapture`/`MemsetDeviceIfCapture` (landed R2/R3b) |

## Tests to port

None upstream. Local gates: the existing TT suite (21/21 default — proves
the env-gated paths are inert), the Qwen3-0.6B/Mistral TT golden pairs
(e2e near-tie when the flag is on), and the capture probe (bisection
instrumentation under `VT_TT_TRACE_DEBUG`).

## Work breakdown

Numbering below is the POST-INVESTIGATION truth (the pre-investigation plan
numbered R2=QkvSplit/RAC device and R3=PA metadata; the bisection showed the
copy/memset/allocation blockers fire FIRST, so those two original items are
now queued behind item 5 rather than being R2/R3).

Each is independently gateable; none blocks another except the capture row,
which wants all three.

### R1 — Device-resident RmsNorm + RoPE at T=1 (threshold flip + perf)

**Problem:** the hybrid thresholds route `RmsNorm` (rows<32) and `RopeNeox`
(T·H<64) to host at T=1. The trace-runner spike measured the perf cost of
flipping them all-device: 12.5 → 10.7 tok/s (~14%, reproduces handoff §6).
Capture must recover that.

**Work:** flip the thresholds to all-device when capture is active (or
unconditionally, gated on `support_static_graph_mode()`), accept the ~1.8
tok/s eager regression, and let capture claw it back. The numerics were
already proven acceptable by `BACKEND-TENSTORRENT-RESIDUAL-GOLDEN`
(device bf16 vs CPU f32 = constant 0.0459 abs, ordinary rounding).

**Sub-blocker:** `RopeNeox`/`RopeFromCache` depend on `BuildCosSinFromPositions`,
which reads `pos` on host (line 1291) and builds cos/sin host-side. The
device RoPE apply path exists (`RopeApplyDeviceNeox`) but the cos/sin
construction is still host. Needs a device-resident cos/sin path OR a
precomputed cos/sin cache uploaded once (the `RopeCosSinCacheKernel` path
already exists for the cache mode — route through it).

**Gate:** op-level `RmsNorm`/`Rope` device parity (already measured); e2e
Qwen3/Mistral gate token-exact or near-tie vs the TT golden.

### R2 — Device-resident QkvSplit + ReshapeAndCache (small host-staged ops)

**Problem:** `QkvSplit` and `ReshapeAndCache` are pure host today — they
read every input via `EnsureHost` and `CommitHost` the output. Both are
bit-exact memcpy/stride ops that went host-staged in W0 because Alloc was
host memory. Inside a captured region they must stay on device.

**Work:** add device-resident variants using `ttnn::slice` (QkvSplit) and
the device paged-write path that already exists for paged KV
(`NotePagedKvRacWrites` / `TryDevicePagedFill` / `TryDevicePagedUpdate` —
landed with residency). The device paged-write path already keeps a ttnn KV
shadow; wire `ReshapeAndCache` to it unconditionally when capture is active.

**Gate:** op-level bit-exactness vs the host path (these are deterministic
copies — byte-identical is achievable and required); e2e gate.

### R3 — Device-resident PagedAttention decode (the big one)

**Problem:** `PagedAttention` at T=1 decode runs the **host f32 oracle**
(`PagedAttentionKernel` host path). The device path
(`TryPagedAttentionDeviceDecode`, `paged_scaled_dot_product_attention_decode`)
exists and is used when the KV shadow is current, but it still reads
`block_table`/`seq_lens`/`query_start_loc` on host (lines 1644-1646) and
reads `query` host (line 1707) before the device call. Those metadata
reads are the capture blocker.

**Work:** keep the metadata tensors device-resident across the decode step
(they are small int32 tensors; upload once per step BEFORE the captured
region, not inside it), and ensure the query entering PA is already device
(no `EnsureHost(query)`). The device SDPA decode path itself is
capture-clean (it's a single ttnn op); the work is removing the host
metadata reads around it.

**Gate:** device PA vs host oracle numerics (already measured: max_abs
~0.0009 for prefill; decode parity measured separately); e2e gate.

### R4 — Flip `support_static_graph_mode()` + wire capture (only after R1-R3)

**Problem:** the platform gate and the `Qwen3DenseDecodeGraph` wiring are
trivial once the region is host-free. This row flips the platform flag,
verifies capture no longer aborts, and measures replay tok/s vs eager.

**Gate:** capture completes (no `TT_FATAL`); replay max_abs=0 vs eager
(already the landed unit-test property); **replay warm tok/s ≥ 12.5**
(the current hybrid eager baseline) — this is the payoff that justifies
all four rows.

### R5 — Make host-free decode the default ([#1604](https://github.com/mudler/vllm.cpp/issues/1604))

**Problem.** R1-R3b and item 5 landed env-gated, and the row's own bar for
the flip is met: the golden re-adjudication landed (#1488 / #1514), and the
same-binary A/B (2026-08-21, P150, head `8399d6121`, batch 1, greedy, warm
legs under one lock) measured 5.1×/5.2× — Qwen3-0.6B 27.1 vs 5.34 tok/s,
Mistral-7B 12.2–13.8 vs 2.35 tok/s. The default path is now the slow one.

**Scope (one change, per #1604).**

1. Default ON through one helper, `vt::tenstorrent::HostFreeDecodeEnabled()`:
   unset or any value except `"0"` → on, exactly `"0"` → off — the
   `vt::GraphCaptureEnabled()` idiom (`src/vt/breakable_graph.cpp:65-71`),
   but deliberately NOT cached (tests toggle the env per case in one
   process). Every `std::getenv("VT_TT_HOST_FREE_DECODE") != nullptr` site
   moves to it (~17 in `tenstorrent_ops.cpp` plus
   `TenstorrentPlatform::support_static_graph_mode()`). The env var survives
   as the A/B opt-out: `VT_TT_HOST_FREE_DECODE=0` reproduces the pre-flip
   default path exactly.
2. **Capture stays declined by default on TT** (measured stop-condition,
   below): `support_static_graph_mode()` additionally requires the explicit
   investigation opt-in `VT_TT_DECODE_CAPTURE`. The captured 27.1 tok/s arm
   is reachable under that env for single-request A/B; flipping THIS default
   is owned by the capture-hang issue.
3. Both device golden pairs go stale the moment the default flips, so both
   are re-captured and re-adjudicated in the same change:
   `qwen3_greedy_0_6b` (refreshed by #1514) and `mistral_greedy_7b`
   (captured 2026-08-12), via `VT_DUMP_IDS` +
   `qwen3-neartie-gap-transformers.py` — the #1488 method, dumped from the
   NEW default (host-free eager) arm. On the post-#1514 main (`52e328789`)
   the flag already anchor-reds the Qwen3 gate fast (prompt[0] tok=1
   engine=14746 vs committed anchor=13); that red is this staleness, not a
   defect claim; the adjudication decides.
4. Concurrency coverage under the new default: both paged-engine gates run
   multi-request and must complete green; the async-serving battery
   (`test_qwen3_dense_async_serving`) runs on TT if its harness selects the
   device — **measured outcome (2026-08-21, P150): 3 FATAL / 5
   checkpoint-absent skip, pre-existing and orthogonal to this flip** — the
   TT backend never advertised async sampled-token readback, so async
   scheduling resolves OFF regardless of decode mode
   ([#1627](https://github.com/mudler/vllm.cpp/issues/1627), under
   `## Owed`); the two flag-pinned default-path cases in
   `test_tenstorrent_backend.cpp` (small-T `kRopeNeox` bit-exact; host-free
   helper inertness) move from `::unsetenv` to `VT_TT_HOST_FREE_DECODE=0`,
   and their meaning becomes "the opt-out path declines". The
   `VT_TT_RECAPTURE_EVERY=8` re-seed arm and the batch-size refusal
   (`VT_CHECK` in `WarmDecodePos`/`WarmPaMeta`/`WarmRacIdx`) keep their
   existing coverage (capture opt-in arms).
5. Records: `docs/BENCHMARKS.md` rows + `.agents/benchmark-record.md` legs
   for the new default vs `=0` on both models, host-free eager (the new
   default), and the captured opt-in single-request leg.

**Measured on the flip tree (2026-08-21, P150, `b86e3705f`):** captured
multi-request decode hangs deterministically — the 16-prompt gate stalls
~10 s into stepping with one tt-metal worker spinning at 100% and the main
thread blocked, both with the plain default and with
`VT_TT_RECAPTURE_EVERY=8`; single-request captured legs (the 27.1 tok/s A/B,
the 80-token gate) never hit it. Host-free EAGER completes the same gate in
35 s (125/125 assertions) and measures 10.94/10.95/11.06 tok/s warm
(Qwen3-0.6B, batch 1, greedy, `--repeat 4`, leg 1 discarded — JIT) vs the
5.34 pre-flip default: a 2.1× default win that does not hang. The captured
5.1× arm is one hang fix away.

**Gates.** Both paged-engine gates green under the NEW default on the
refreshed pairs; re-adjudication max gap within the 500-mnat band, zero
cells outside top-K (the #1488 bars); `test_tenstorrent_backend` green with
and without an ambient opt-out; the A/B re-run on the flip tree showing the
default leg at the host-free eager rate.

**Stop conditions.** A non-near-tie divergence under the new default stops
the flip and becomes the work. A hang or a refusal firing on an ordinary
serving dynamic stops the flip until that path is fixed or refuses loudly —
the captured-arm hang is held out of the default by scope item 2 and owned
by its issue; #1105's `DecodePosCache` identity fix stays owed.

## Dependencies

```
R1 (RmsNorm+RoPE device)  ─┐
R2 (QkvSplit+RAC device)  ─┼─► R4 (capture wire + measure) ──► decode tok/s win
R3 (PA decode metadata)    ─┘
```

R1, R2, R3 are independent and parallel-claimable. R4 is the integration
row that wants all three + produces the headline number. If R4's replay
tok/s does NOT beat 12.5, the whole effort is a wash — but that can only be
known after R1-R3, which is the cost of answering it.

## Gates (per row + integration)

- **Correctness:** every device-resident variant must be bit-exact or
  near-tie vs the current host path, gated by the existing TT golden pair
  (`our_ids_tenstorrent.npy` / `neartie_gap_mnats_tenstorrent.npy` for
  Qwen3-0.6B, the Mistral pair for Mistral-7B). RED-first op-level test
  before each e2e gate.
- **Capture (R4 only):** `TT_FATAL`-free capture + replay max_abs=0 +
  replay warm tok/s ≥ 12.5 (Qwen3-0.6B `vllm-cli` smoke, same harness as
  the trace-runner spike).
- **No perf regression outside capture:** the threshold flips in R1 regress
  *eager* tok/s (12.5→10.7) — that regression is acceptable ONLY because R4
  recovers it. If R4 is not landed, R1 must not ship unconditionally; it
  must gate on `support_static_graph_mode()` so non-capture runs keep the
  hybrid thresholds and the 12.5 baseline.

## Risk

- **R3 is the scope risk.** R1 and R2 are mechanical (flip + reuse existing
  device paths); R3 (device PA decode with device-resident metadata) is
  real work and the most likely place to find another host touch.
- **R4's payoff is uncertain until measured.** The whole plan exists to
  answer "does capture beat 12.5 tok/s"; if it doesn't, R1-R3 still
  delivered device-resident ops (useful for future prefill capture) but no
  decode win. That's an honest outcome, not a failure — it's the
  measurement the trace-runner spike owed and couldn't make.


### Landed (this branch, measured on P150)

- R1 threshold flip (RmsNorm residual + PreferDeviceRope all-device under
  the flag).
- R2 `CopyDeviceDeviceIfCapture` (ttnn::empty + ttnn::copy device->device).
- R3 program-cache warm (`enable_program_cache` + eager-warm of the copy ops).
- R3b `MemsetDeviceIfCapture` (ttnn::zeros into the existing shadow).

### Open (item 5 — the payoff port)

Persistent device input tensors in the decode-graph slot + populate before
capture/replay via `ttnn::copy_to_device` (never inside capture). Ported
from the tt-metal vLLM plugin (see Port map).

### Queued behind item 5 (from the original plan; may or may not be needed)

Device-resident QkvSplit + ReshapeAndCache variants, and PA decode with
device-resident metadata. The bisection has not reached these (the
enqueue_write fatal fires first); keep or drop them per what item 5's probe
surfaces.

### Known constraints of the investigation code (env-gated, carried forward)

Recorded from review; all are flag-gated-only and acceptable for an
investigation row but MUST be addressed by the item-5 port:

1. `CopyDeviceDeviceIfCapture` ignores `bytes` — a partial/interior Copy
   between two same-sized shadowed slots clones the WHOLE src shadow.
2. It does not update `dev_rows`/`dev_cols`, so a consumer view matching
   the logical shape but not the recorded shadow shape can fall into an
   EnsureHost re-upload (a readback during capture — defeating R2).
3. The equal-BYTE check does not pin dtype/shape (a same-byte bf16/f32
   reinterpret is possible).
4. `enable_program_cache()` fires inside the copy helper; if the captured
   region never takes that path it is never enabled. Belongs in
   TraceBeginCapture (or platform init) for the port.
5. The `tt_capture_active()` clear is not exception-safe (a throwing
   end_trace_capture leaves it stuck true, flipping eager Copy/Memset).
   The inertness guard test catches the stuck-true case; the port should
   make the clear RAII.
6. TOCTOU on SlotMutex around the ttnn calls (re-acquire without
   revalidating the slot).
7. `d->device = std::move(cloned)` drops the prior dst shadow mid-capture
   (a dealloc during a live trace).

## Owed

- **Qwen3-4B TT forward beyond the near-tie band on both arms
  ([#2811](https://github.com/mudler/vllm.cpp/issues/2811)).** Found by the
  #1625 flip's 4B bring-up (2026-09-03): deterministic on both arms, but
  5/16 prompts past the 500-mnat band against teacher-forced gaps on each
  arm's own prefix (capture worst 1000 mnats @ (11,3), eager worst 2000 @
  (5,6)). The 4B dense gate skips loudly on TT until the lane lands a passing
  pair or a falsified-drift root cause. The two 4B capture FATALS found on
  the way are NOT owed — fixed in commit `1872aeb16` (shadow-volume
  contract).
- **Qwen3.5-GDN captured arm: structurally blocked mid-capture; the
  device-pure GDN wave is owed
  ([#2907](https://github.com/mudler/vllm.cpp/issues/2907), found by
  [#2812](https://github.com/mudler/vllm.cpp/issues/2812)).**
  RE-CORRECTION (2026-09-04, P150 repro 2/2 deterministic): the previous
  CORRECTION on this bullet was WRONG — the original "Qwen3.5-GDN
  TT_FATALs mid-capture" record was right. Its supporting M2 mutation
  almost certainly hit the INERT MoE driver site (`Qwen3_5DecodeGraph`),
  not the dense driver (`Qwen3_5DenseDecodeGraph`) the 0.8B checkpoint
  exercises, so its 16/16 green measured the eager default, not the
  captured arm. The captured 0.8B battery dies deterministically on
  tt-metal "Writes are not supported during trace capture"
  (fd_mesh_command_queue.cpp:760). Root cause: three per-call staging
  classes inside `Qwen3_5DenseDecodeGraph::Step`. (1) The `RmsNormKernel`
  gemma branch re-uploaded the baked w+1 f32 buffer every call — FIXED
  (a `gemma_device` slot on `BufferSlot` caches it). (2) `MatmulBT` weight
  interior views — the non-merged GDN path Slice's `packed_weight` per
  call, which `EnsureDevice2D` refuses (W2c) — fell to anonymous
  `from_span` uploads every call — FIXED (a weight-view shadow cache keyed
  by view pointer and geometry; tracked bases keep `EnsureDevice2D`).
  (3) The GDN ops themselves (`CausalConv1dUpdateKernel`,
  `GdnDecodeKernel`) are host-orchestrated BY DESIGN — EnsureHost on
  x/q/k/v/g/beta, ReadIdxHost, per-call host vector builds, per-call
  UploadTensor — which no trace capture can admit; removing that is the
  device-pure GDN wave, a separate row and issue. Blindness: the ambient
  q35 gate still adjudicates only the eager arm and no captured pair gate
  exists; a conjunct's deletion for q35 now fails loudly (the captured
  battery hits the deterministic fatal) rather than passing silently, but
  the arm stays ungated until the wave lands.
- **Mistral-7B captured pair LANDED (2026-09-04, #2812);
  `MistralForCausalLM` joins the default-on arch set.** The captured arm
  is byte-identical across two runs with a card reset between, and differs
  from the committed eager pair in 34/256 cells spanning 5 prompts (worst
  p1 t8-t15 — the same p[1] t=8 cell the M1 mutation flagged), so
  #2566's CLI-level token identity does not extend to the battery. The
  pair is teacher-forced via transformers: every cell sits inside the
  500-mnat band, worst 250 mnats at (1,8). The captured gate greens
  128/128 against its own pair, and the ambient mistral gate now
  adjudicates the CAPTURED arm against the capture pair. Llama and
  InternLM2 keep the explicit opt-in until their own captured arms are
  brought up.

- **The capture arm's cold step emits a deterministic wrong first decode
  token ([#2461](https://github.com/mudler/vllm.cpp/issues/2461)) — REPAIR
  LANDED in this change (2026-09-01, root cause and evidence in `## Now`);
  the issue closes on merge. RESIDUAL, a separate defect, now filed and
  repaired: the capture-armed battery's one remaining failure at prompt[1]
  tok=1 is cross-request persistent shadow state, not the slab swap
  ([#2469](https://github.com/mudler/vllm.cpp/issues/2469), REPAIR LANDED
  in this change — see its own `## Owed` entry below).** Found by
  the wave's fresh mutation review (2026-09-01): the reviewer's rerun of the
  capture-armed focused gate printed deterministic gibberish where the
  operator's evidence record claimed "coherent" — the gate's `words>20`
  criterion was tt-metal log lines on stdout, and the A/B/C capture-leg logs
  hold the same bytes. Per-step adjudication (`VT_TT_DUMP_KV`): prefill
  exact, COLD step wrong (48755 vs plain-eager 3364 " story" on the same
  binary and inputs, top-2 gap 1.69 — not a near-tie), capture step and
  replays then consistent on the corrupted history. The capture-armed golden
  battery's single 1/32 failure (prompt[0] tok=1) is the same defect — it
  was mis-filed as a near-tie earlier in this record. Replays are
  token-exact (battery prompts 1-15), so the defect is the driver's eager
  cold pre-warm step (`qwen3.cpp:1086-1111`) under the Warm*-staged
  cached-device-tensor paths leg A never exercises; driver-side attn
  metadata on that step is verified correct; the root cause is the
  stride-fabricated shadow descriptor (`## Now`). Pre-exists both R4
  fixes (fix 1 is warm-branch-only, fix 2's fresh-slot arm is
  capture-only). R5-era latent: no token gate ever ran on the captured
  CLI path. The repair carries red→green evidence and a token-clean
  A/B/C re-measurement (`## Now`); the machinery-only caveat on the
  earlier captured-arm ratios is superseded by it.
- **The recycled same-size slot replays the previous request's final decode
  position ([#2469](https://github.com/mudler/vllm.cpp/issues/2469)) —
  REPAIR LANDED in this change (2026-09-01, root cause and evidence in
  `## Now`); the issue closes on merge.** A same-size request boundary
  under capture kept the previous request's final `cur_pos` on the shared
  device tensor, so PA attended the dead request's decode KV rows and RAC
  wrote at the stale virtual position, decoding the previous prompt's
  tokens. The repair adds the per-slot continuation predicate and
  `expected_cur_pos` bookkeeping, routes a same-size boundary through the
  #1476 re-capture lane (seed-only measured insufficient: 13/11/264 with
  device state verified correct), and makes `WarmPaMeta`'s `cp_host` read
  the `DecodePos` host mirror so the device-PA guard reads what the device
  actually holds. Green: capture-armed battery 125/125 assertions, 16/16
  prompts PASS, max gap 0.375 nats @ prompt[1] tok=1 (inside the ≤0.5
  near-tie band), 1344 device-PA selections / 0 declines; default arm
  unchanged (125/125); captured CLI deterministic and coherent across a
  card reset. Residual near-tie at that cell stays within the row's
  adjudication band.
- **Continuous-batching blind spot: the continuation predicate reads only
  `seq_lens[0]` — a mid-batch request replacement on row r>0 at constant
  num_reqs (continuous batching) leaves `cur_pos[r]` stale and the predicate
  blind.** Unexercisable by the single-slot battery; same defect class as
  #2469. The continuation predicate `seq_lens[0]-1 == expected_cur_pos`
  covers row 0 / the single-request slot the capture path serves today;
  multi-row decode request swap is unexercised and unsolved.
- **The PA decline counter does not count guard-thrown declines (the `PA meta
  not warmed for this step` fallback path).** Wiring it in measured 6720
  declines on the default arm's battery vs the asserted 0, so it owes its
  own change with the test's expectation re-baselined, owned by this row
  (2026-09-01).
- **The default-polarity question reopened by
  [#2003](https://github.com/mudler/vllm.cpp/issues/2003): RESOLVED as a
  documented stand-pat (2026-08-30, closed by the W2 record PR).** The
  confirming A/B under the W2 sampler reproduced the inversion at clock
  parity: default eager median 10.998 tok/s vs 14.299 for
  `VT_TT_HOST_FREE_DECODE=0`, ratio 1.300 median / 1.294 mean, judge PASS,
  every busy slice of both arms exactly {1350} MHz with the cap read from
  firmware (`.agents/benchmark-record.md`, 2026-08-30 entry). The default
  stays host-free. STILL OWED on this row: name the corrected mechanism for
  the opt-out arm's ~2.5x improvement. The EnsureDevice2D hypothesis is
  REJECTED — the per-op delta `b86e3705f..21fe11cf1` shows `353511e72` /
  `101b415d7` only ADD staging work on the hybrid path, which cannot explain
  the movement while the default arm is unchanged — so until the corrected
  mechanism is named, the inversion is a real, unexplained performance
  property of the shipped polarity, and a flip would be a guess.
  (2026-08-31: a second candidate — per-step overhead from the default arm's
  failed device-PA attempts — is REJECTED by measurement: the opt-out probe
  shows the identical regime, 896 PA attempts / 868 failures / 0 device RAC /
  0 warm hooks, under `VT_TT_HOST_FREE_DECODE=0`; see `## Now`. The
  mechanism stays unnamed.)
- **No case pins `HostFreeDecodeEnabled()`'s no-caching contract on the RAC
  path ([#1688](https://github.com/mudler/vllm.cpp/issues/1688)).** The R5
  fresh review found `ReshapeAndCacheKernel` still latching the flag in a
  function-local `static`, which after the default flip cached ON and stripped
  `VT_TT_HOST_FREE_DECODE=0` of its effect on that path for the rest of the
  process. The latch is FIXED IN FLOW (live read, matching every other
  converted site), but no test reds on it: `ReshapeAndCacheKernel` needs a real
  Blackhole device, so every case that reaches it sits behind
  `TenstorrentPresent()` and skips on every host in the `rc` fleet. Closing it
  means a `thalia` case that drives the RAC path twice across a
  `setenv`/`unsetenv` of the flag and asserts the second call takes the other
  branch — the same shape the flip already used for
  `support_static_graph_mode`, which could be written host-side because the
  platform accessor needs no card.
- **Captured multi-request decode hangs ([#1625](https://github.com/mudler/vllm.cpp/issues/1625)).**
  Measured on the R5 flip tree: the 16-prompt gate hangs ~10 s into stepping
  under captured decode, with `VT_TT_RECAPTURE_EVERY=8` alike, while
  host-free eager completes in 35 s (11.0 tok/s warm). R5 therefore ships
  `support_static_graph_mode()` declined by default (opt-in
  `VT_TT_DECODE_CAPTURE`); the captured 27.1 tok/s arm stays one hang fix
  away, and that fix owns flipping this default back.
  (2026-09-01, this branch: boundary SHARPENED — same-prompt sequential
  generates are clean under capture (`--repeat 2`, 45 replays, rc 0; the
  A/B/C capture legs ran `--repeat 5` in-process, 474 replays each), while
  the 16-prompt battery spins in pure userspace after one capture. The
  trigger is a cross-prompt KV-geometry re-warm under a live trace, and
  tt-metal spins (state R, stime=0) instead of the R5-era futex sleep.
  (2026-09-01, later: the capture-armed golden battery itself COMPLETES at
  tip — 31/32 exact, the single failure is #2461's cold-step token, not a
  hang; what still spins is the `VT_DUMP_IDS` battery dump. Blocked on this
  fix: the capture-arm near-tie pair for `qwen3_greedy_0_6b`. The
  capture-default flip is blocked on BOTH this fix and #2461.)
- **TT never advertises async sampled-token readback
  ([#1627](https://github.com/mudler/vllm.cpp/issues/1627)).** The
  async-serving battery FATALs on TT at its anti-vacuous-pass guard
  (`REQUIRE(loaded->async_scheduling_enabled())`,
  `test_qwen3_dense_async_serving.cpp:124`) for every cached checkpoint —
  3 FATAL / 5 checkpoint-absent skip on the P150 — because
  `vt::Backend::SupportsAsyncSampledTokenReadback()` has no TT override, so
  async scheduling resolves OFF (`max_concurrent_batches=1`). Pre-existing at
  base `52e328789` (zero hits under `src/vt/tenstorrent/`) and orthogonal to
  the decode-mode flip; enabling it needs a device-mirrored sampled-id design
  (the CUDA `async_device_mirror` equivalent against the tt-metal allocator)
  and owns the battery going green on TT.
- **`DecodePosCache` is keyed on bare `num_reqs`, with no engine, queue or device
  identity, and is never cleared.** Two engine instances in one process at the same
  padded batch size therefore share one `cur_pos` device tensor: the second engine's
  first `WarmDecodePos` finds the first engine's entry, returns early, and both
  `WarmRacIdx` / `WarmPaMeta` aliases bind to a buffer another engine is advancing.
  That is silently wrong rather than a refusal, and it is a candidate explanation for
  `test_qwen3_paged_engine` still timing out under the flag. Owned by
  [#1105](https://github.com/mudler/vllm.cpp/issues/1105).
- **`VT_TT_RECAPTURE_EVERY` lagged `cur_pos` by one per recapture cycle; the
  mechanism is fixed by [#1476](https://github.com/mudler/vllm.cpp/issues/1476).**
  The regime flag no longer comes from the process-global `GraphCapturesDone()`
  (never cleared by `Reset()`); the driver passes `s.graph.captured()`, so the
  cold eager step after ANY reset — a block boundary or a
  `VT_TT_RECAPTURE_EVERY` cycle — re-seeds `cur_pos`, and the RAC page_table
  refreshes on content change. The deeper defect stays open on
  [#1105](https://github.com/mudler/vllm.cpp/issues/1105): `DecodePosCache` is
  keyed on bare `num_reqs` with no engine or device identity, and the real fix
  is a per-cache-entry seed / generation field aliased on every warm call, not
  any process-global counter. The eager PA consistency check remains
  self-validating (`e.cp_host` against the `seq_lens` that wrote it). The
  re-seed regime now has a gate arm: `VT_TT_RECAPTURE_EVERY=8` forces
  mid-generation re-captures (9 captures over 80 tokens), byte-identical to
  the plain captured arm, and the `GraphCapturesDone()`-early-return mutation
  reds it (see `## Now`).

The seven constraints above remain. A new batch size after the first
capture is now refused (`VT_CHECK` in `WarmDecodePos` / `WarmPaMeta` /
`WarmRacIdx`) rather than freezing `cur_pos`. The real fix is a
per-cache-entry seed / generation field and aliasing on every warm call,
not a process-global `GraphCapturesCounter`. Tracked on
[#1105](https://github.com/mudler/vllm.cpp/issues/1105).

- **The `retired_pts` keep-alive is unguarded defense-in-depth.** No gate
  exercises it and none can on the current fixed-width setup: the engine
  preallocates `block_table_num_cols=256`, so after the first allocation no
  width change (growth or shrink) ever occurs and the retire branch in
  `WarmRacIdx` is structurally unreachable there. A width change needs a
  multi-request run whose longest request finishes (shrink) or a driver that
  grows cols per block (growth) — neither is reachable through the current
  decode-graph gate. It stays because freeing a device buffer a recorded
  trace addresses is the worse failure; treat it as defense-in-depth until a
  gate can reach it, and do not cite it as covered. The allocator itself now
  handles ANY width change (`!=`, growth or the multi-request shrink when the
  longest request finishes — the old `>` let the else-branch
  `copy_to_device` TT_FATAL on a shape mismatch), mirroring the driver's
  `cols_changed !=` reset.
- **RESOLVED 2026-08-27 — `test_tenstorrent_backend` exited 139 after a fully
  green doctest summary** (23/23 cases, 831/831 assertions): static
  `std::optional<ttnn::Tensor>` cache fields are destroyed after the UMD device
  closes and `deallocate_impl` reaches a torn-down `GraphTracker`. Proven
  pre-existing at `origin/main` by an A/B stash/build/run during the
  [#1476](https://github.com/mudler/vllm.cpp/issues/1476) gate. Fixed by
  [#1486](https://github.com/mudler/vllm.cpp/issues/1486): every process-lifetime
  TT cache in `tenstorrent_ops.cpp` is now heap-allocated and deliberately never
  destroyed, so no `ttnn::Tensor` destructor can run after tt-metal teardown.
  Suite and e2e binaries exit 0 (mutation: reverting one accessor re-adds the 139).
- **`test_release_metadata` is red on every aarch64 host**: the fixture stages
  the host `/bin/true` into an `x86_64`-named archive, so `agent-preflight`
  cannot go green on the TT dev fleet. Found while running this row's preflight.
  Owned by [#1487](https://github.com/mudler/vllm.cpp/issues/1487).
- **The TT `test_qwen3_paged_engine` golden was stale** — RESOLVED 2026-08-20
  in this flow: the anchor drift prompt[1] tok=10 (engine=14126, committed
  6290 — the logged `62901` was a print artifact, #1508) came from the default
  decode path having moved since the 2026-08-10 capture. The before/after-#1476
  comparison recorded in #1488 compared exit codes and only the p1-tok-10
  engine token, so it does not establish that pre-fix TT matches the refreshed
  cells (prompt 5 tail, prompt 7); the golden is derived from this branch's TT
  output and the refresh stacks here. Re-adjudicated with `VT_DUMP_IDS` (eager
  `VLLM_CPP_CUDAGRAPH=0` and captured dumps byte-identical, md5
  `b5307e33…`) and `qwen3-neartie-gap-transformers.py` (transformers 4.57.1,
  torch 2.10.0a0+cpu): 53 cells refreshed across 7 prompts — single
  near-tie divergences at one token each (p1 tok10, p5 tok10, p10 tok5,
  p11 tok4, p12 tok13, p15 tok12) whose greedy continuations then follow
  the new prefix, plus prompt 7 rewritten from tok0; the new p7 row
  matches the vLLM greedy sequence exactly, max gap 375 mnats, zero cells
  above the 500-mnat band, zero outside-top-K. Fixed by
  [#1488](https://github.com/mudler/vllm.cpp/issues/1488).
- **doctest `MessageBuilder` streams `const char*` as bool**: every
  separately-bound `const char*` in a `MESSAGE`/`REQUIRE_MESSAGE` renders as
  `1`, so the anchor-drift message printed `committed anchor=62901` for a
  golden holding `6290` (and `96251` for `9625`), and `label` printed as `1`
  instead of `qwen3-0.6B`. Reproduced against the pinned header with a
  7-line harness; fixed by passing `std::string` in
  `test_qwen3_paged_engine.cpp`. Found during the #1488 re-adjudication after
  the garbled value had been misread as golden-buffer corruption. Fixed by
  [#1508](https://github.com/mudler/vllm.cpp/issues/1508).
- **Device-PA decode consumes the KV shadow on `device_current` alone
  ([#2670](https://github.com/mudler/vllm.cpp/issues/2670)).** Latent after
  #2669's repair removes the one known trigger: the reader-side contract
  has no proof besides the flag, so any future publisher over a partially
  correct device block corrupts decode with no error path. Repair
  direction: a per-block coverage stamp the push records and the reader
  checks before it skips the upload; mirror upload stays the fallback.
- **`VT_DUMP_IDS=1` turns the anchor REQUIRE off and the verdict line does
  not say so ([#2671](https://github.com/mudler/vllm.cpp/issues/2671)).** A
  dump-mode run prints `16/16 prompts PASS` from the committed goldens
  alone, which is how a build that reds 14 of 16 prompts outside dump mode
  looked green on 2026-09-02. Repair direction: mark the verdict
  `RE-CAPTURE MODE` and report skipped anchors; keep the
  `qwen3-neartie-gap.py` refresh path working.

The operator must still rerun the 80-token no-hang gate and
`test_qwen3_paged_engine` on a Blackhole P150. An implementer run is an
input, not a gate result.

## Now

`ACTIVE`. R1-R3b and the R2 on-device `cur_pos` / `update_idxs` advance are
implemented on this branch, env-gated by `VT_TT_HOST_FREE_DECODE`.

### Capture-default flip staged (2026-09-03)

The #1625 wave is specced under `## Capture-default flip` below: capture
becomes the DEFAULT (`VT_TT_DECODE_CAPTURE=0` opts out), the Qwen3-4B
captured pair is brought up in the same change, the published benchmark
figure is re-taken on the flip tree, and the flip lands stacked on the #2669
repair once PR #2672 merges.

### #2812 pairs lane (2026-09-04)

Mistral-7B's captured pair is committed and its captured arm gates green
against it (teacher-forced, worst 250 mnats); `MistralForCausalLM` joins
`DecodeCaptureDefaultArch`, so the ambient mistral gates adjudicate the
CAPTURED arm. The Qwen3.5-0.8B captured arm stays owed: it dies
deterministically mid-capture on the host-orchestrated GDN ops; the
per-call gemma-norm and weight-view staging classes found on the way are
fixed on this branch, and the device-pure GDN wave owns the rest
(#2907). See `## Owed`.

### Repair (2026-09-03): short-chunk device KV push clobber (#2669)

The captured multi-request battery reds at the first cross-request KV block
boundary. The boundary decode step emits deterministic punctuation garbage
(the 11/13/264 family) while eager host-free stays green; #1625 carries the
symptom. Root cause, probed on the P150 with scratch instrumentation that
never landed:

- `TryDevicePagedPushPair` routes a prefill chunk shorter than
  `kPagedFillMinTokens` (16) to `TryDevicePagedUpdateBatch`. The batched op
  treats each chunk token as a separate batch user over a synthetic
  one-entry page-table stick. All users of one chunk resolve to the same
  physical block, tt-metal `paged_update_cache` is a page-granular
  concurrent read-modify-write, so the users clobber each other and the
  last writer wins. The device block keeps the previous request's rows
  0..3, patches only the final row, and leaves the rest stale, while the
  push site publishes `device_current = true`.
- The boundary decode step reads `sk.device_current` and consumes the
  device shadow without a re-upload, so device-PA attends the dead
  request's KV rows. Request 0 is always clean: its prefill push declines
  (`can_update` is false with no shadow yet), so the mirror re-uploads.

Evidence: the device-vs-mirror diff at the boundary shows K maxdiff 54-342
with rows 0..3 byte-identical to the dead request's values and rows 5-6
stale nonzero against a zeroed mirror; the virgin-step control diff is
0.000000. The flag history shows five mirror patches then `prefillpush OK
B=5` per layer on the second request's prefill. The fix probe (fill at any
T, threshold 16 to 1) moves the failure from prompt[1] tok=1 hard garbage
to tok=14 deterministic near-tie.

Re-measured at tip `4d10c8acc` (2026-09-03, uninstrumented): the DEFAULT
eager arm stays anchor-exact — the SACRED battery is 16/16 PASS with the
committed goldens, so the pair stays valid and no default-arm refresh rides
this repair. The CAPTURED arm reds the anchor REQUIRE at prompt[1] tok=1
(engine 30, committed 572); which wrong token appears moves run to run
(374 in the probe session, 30 here), which is the race, not a different
defect. The trigger is capture-only: the eager arm never consumes the
stale shadow.

Plan, in order, one pull request: (1) commit this spec; (2) a red-first
focused gate over prompts 0 and 1 that keeps the anchor-exact REQUIRE and
runs under `VT_TT_DECODE_CAPTURE=1` — it reds at prompt[1] tok=1 before the
repair; (3) the repair: route a sequential fill-eligible chunk to
`TryDevicePagedFill` at any T, or refuse the batched-update path when two
chunk users share one physical block; (4) the full gate on the P150: the
focused gate, the SACRED default-arm battery, and the captured battery;
(5) pin the captured arm with its own committed golden pair
(`our_ids_tenstorrent_capture.npy` / `neartie_gap_mnats_tenstorrent_capture.npy`),
dumped from the repaired tree and teacher-forced with the #1488 method
(`qwen3-neartie-gap-transformers.py`, transformers 4.57.1 CPU) — the same
method that refreshed the default pair at #1630. Post-repair the captured
sequence resolves one near-tie differently from the eager anchor (probe:
tok=14), so the captured arm cannot share the eager pair. (6) the records:
#2669 closes on merge, #2670 and #2671 ride as Owed. #1625's
capture-default flip stays blocked on this repair.

LANDED 2026-09-03 (commit `7ee345ef5`, repair = threshold 16 to 1 in
`TryDevicePagedPushPair`/`TryDevicePagedPush`; #2669 closes on merge).
Post-repair the captured arm's first divergences from the eager anchor sit
at p1 t14, p2 t1, p6 t12, p7 t2, p9 t2, p10 t11, p12 t13 and p13 t7, each
followed by that prompt's own continuation; the boundary cells themselves
(p1 t0..t4) match eager exactly, so the clobber is gone, and the residual
is the captured-vs-eager near-tie class #1476 recorded. The captured pair
was dumped from the repaired tree byte-identical across two runs with a
card reset between, then teacher-forced: 18 of 256 cells carry any gap, max
500 mnats, zero cells outside top-K, 238 of 256 cells the teacher's exact
argmax on our prefix. Teacher environment drifted from the #1488 record:
transformers 5.16.1, torch 2.13.0+cu130 on CPU (the 4.57.1 environment no
longer exists on this host); the oracle registry's sub-ULP caveat cannot
reach this pair because the instrument's quantization error sits two orders
below every certified gap. Green on the P150, one card reset per run: eager
battery 16/16 anchor-exact unchanged (max 0.375 nats, rc 0), captured
battery 16/16 against the new pair (max 0.5 nats, rc 0), focused capture
gate 2/2 (rc 0), `VT_TT_RECAPTURE_EVERY=8` captured battery 16/16 (rc 0,
the re-capture lane tolerates the fill path), `test_tenstorrent_backend`
52/52 cases 5983/5983 assertions. A fresh reviewer returned PASS on the
review range `77224426e..7ee345ef5`: reverting the threshold to 16 reds the
focused capture gate at prompt[1] tok=1 (engine 11 against the committed
572; the wrong token differs from the spec's 30/374, consistent with the
race), corrupting a captured-pair cell reds it again naming the corrupt
value, the eager SACRED battery stays 16/16 green (max 0.375 nats), and the
reachability mutation, `can_update=false` in `NotePagedKvRacWrites` so both
device push call sites die, greens as expected, which pins the M1 red to
the production push site. Statically both push functions have exactly one
caller each, both in `NotePagedKvRacWrites`
(tenstorrent_ops.cpp:1175,1187), whose only caller is the production
kReshapeAndCache path (tenstorrent_ops.cpp:3105); no test-only path exists.
Mutation logs live in `/tmp/review-2669-logs/` on the gate host.

The #2566 rate figure survives the repair, re-taken on this head
(2026-09-03, P150, the #2566 recipe: order-alternated triples,
`--repeat 5` with leg 1 discarded, warm medians over 12 legs, one flock
per batch, card reset first, harness `~/hf-r2672-gate3.sh`, raw logs
`~/hf-r2672-t{1,2,3}{A,B,C}.{out,err}`): captured 28.61 tok/s against the
27.47 pre-repair record, default 12.21 against 12.90, opt-out 15.61
against 17.80 - that arm's band is the unclosed inversion residual the
#2566 entry already records, and the R5-era 5.34 figure bounced to 17.80
before it. Capture over default 2.34x, over opt-out 1.83x. Zero fatals,
zero hangs, 470 replays on every capture leg. The repair costs the
captured arm nothing, and the payoff figure the capture-default flip
stands on is measured on the repaired tree.

The operator gate (2026-08-20, P150, `206afb63`) found
[#1476](https://github.com/mudler/vllm.cpp/issues/1476): captured replay went
degenerate at the first KV block boundary while host-free eager stayed
coherent — the recorded 22/22 argmax predated the final `cur_pos` integration
and did not reproduce on the landed tree. Root causes, both fixed in this
change: the RAC `page_table` was `[C,1]` where the tt-metal reader indexes the
full stick (`page_table_ptr[update_idx / block_size]`), so every write past
block 1 landed in a garbage physical block; and `WarmDecodePos` keyed its
skip on the process-global `GraphCapturesDone()`, which `Reset()` never
clears, so a post-boundary re-capture read `cur_pos` one position behind.

Implementer verification on the P150 (2026-08-20, this change, full-answer
compares — never a `grep -m1` first-line artifact):

- **The #1476 degeneration is gone.** Reverting either root cause in a
  /tmp scratch clone (same TT build config) regenerates it: the G1 mutation
  (page_table back to `[C,1]`, steady-state refresh removed) keeps 32 steps
  of argmax agreement then reds at step 33 — the first step past
  `block_size=32` — with a non-tie divergence (argmax 1555 gap 0.75 vs
  eager 13) and the word salad ("straight line line line on road…");
  the G2 mutation (`GraphCapturesDone() > 0` early-return) reds the
  recapture arm at step 11, the first step of the second capture cycle.
  Both restores are sha256-verified byte-for-byte, rebuilt, and rerun
  green with answers byte-identical (284B md5 `3b5a579d…`) to the
  worktree gate runs. A 160-token captured run is coherent and the
  80-token captured answer is a strict byte-prefix of it (5 block
  boundaries crossed).
- **Captured vs host-free eager is NOT byte-identical** — the prior
  byte-identical claim was a first-line compare artifact. Full answers:
  captured 284B md5 `3b5a579d82d58396fe4e344826946403`, eager 286B md5
  `f5ffdf6aa290e11fd187673c2f3c52bb`, first diff at byte 174, both arms
  coherent. Adjudicated per-step (`VT_TT_DUMP_KV` top-2 dump; the top-2
  raw-logit gap is the logprob gap in nats, the `qwen3-neartie-gap.py`
  bar): argmax identical for 45/80 steps with top-2 values agreeing to
  ≤0.5 logits (≤4 bf16 ULP at the ~20-logit scale); the FIRST divergence,
  decode step 46, is a swapped top-2 near-tie — captured `[11:19.75,
  311:19.50]` gap 0.25 nats vs eager `[311:19.625, 11:19.50]` gap 0.125
  nats (1-2 bf16 ULP), cross-arm deltas at the tied pair 0.125/0.25 —
  inside the near-tie band this repo already tracks for Qwen3-0.6B on TT
  ([#1488](https://github.com/mudler/vllm.cpp/issues/1488) owes the
  teacher-forced golden re-adjudication). The 34 argmax differences after
  step 46 are prefix divergence (each arm greedy-decodes its own prefix),
  not numeric evidence.
- **The `cur_pos` re-seed regime (G2) is gate-covered.**
  `VT_TT_RECAPTURE_EVERY=8` forces 9 captures / 71 replays (8
  destroy+re-capture cycles mid-generation); that arm is byte-identical to
  the plain captured arm (same 284B md5) and carries the same single
  step-46 near-tie vs eager. Restoring the old
  `GraphCapturesDone() > 0` early-return in a scratch build reds this arm
  at step 11 (non-tie divergence, incoherent text); restoring the fix
  greens it. Without the arm the fixed-width 80-token gate never fires a
  `Reset()` (the engine preallocates bt_cols=256), so exactly 2 re-seeds
  run and the guarantee was undetected.

`test_tenstorrent_backend` 23/23 + 831/831 green with and without an
ambient `VT_TT_HOST_FREE_DECODE` (its exit-time segfault is pre-existing,
[#1486](https://github.com/mudler/vllm.cpp/issues/1486)).

Next: operator rerun of the 80-token captured-vs-eager gate and
`test_qwen3_paged_engine` on card; the paged-engine golden re-adjudication is
done ([#1488](https://github.com/mudler/vllm.cpp/issues/1488) closed by the
stacked golden-refresh commit on this branch).

#1498 and #1514 merged (2026-08-21, `d27639e71` / `49c64bbc8`), closing the
row's recorded prerequisites for the flip. R5 is now `ACTIVE` on
`row/BACKEND-TENSTORRENT-HOST-FREE-1604` from main `52e328789`. The
under-flag reproduction on that base (2026-08-21) is recorded under R5 scope
item 2: fast anchor red, no timeout.

**R5 implemented and gated on the branch (2026-08-21, commits `fdedfee12`
spec, `b86e3705f` flip, `f85492992` capture-decline #1625, `9a7d9f4d4`
golden refresh #1626/#1627).** All R5 gates green on the P150: both
paged-engine gates 16/16 under the NEW default on the refreshed pairs
(Qwen3 125/125 assertions, max gap 0.375 nats; Mistral 128/128 assertions,
max gap 0.25 nats, 0 forward-divergent both, Mistral exit-139 is the #1486
teardown class after the doctest SUCCESS); re-adjudication max gaps 375
(Qwen3) / 250 (Mistral) mnats, zero cells outside top-K;
`test_tenstorrent_backend` 23/23 + 831/831 with and without an ambient
opt-out; default leg 10.94/10.95/11.06 tok/s vs the 5.34 opt-out (2.1x,
same-binary A/B). The async-serving battery outcome is pre-existing and
filed (#1627, under `## Owed`). Pending: fresh review, PR, operator merge.

**R4-at-tip wave opened (2026-08-31, worktree
`row/BACKEND-TENSTORRENT-HOST-FREE-FORWARD` @ main `6a544bdb8`).** Operator
gate rerun on the merged tip, all under one `flock $HOME/gpu.lock`:

- Golden gate, default arm, this P150: PASS — 16/16 prompts, 125/125
  assertions, max gap 0.375 nats, 0 forward-divergent; device-op proof
  0 declines (`kPagedAttention selections=7168`).
- Same-binary A/B (3 order-alternated pairs, `--repeat 5`, discard run 1,
  warm medians): default 12.92 vs `VT_TT_HOST_FREE_DECODE=0` 17.78 tok/s —
  the inversion widened to 1.376x after W2c/#1476 (1.300 on 2026-08-30).
- Capture never armed in any default-arm leg. Root-cause chain, measured
  with `VT_TT_TRACE_DEBUG=1`: `VT_TT_DECODE_CAPTURE` unset keeps
  `support_static_graph_mode()` false (`platforms/tenstorrent.cpp:85-88`),
  so `Qwen3DenseDecodeGraph::Step` never runs (`qwen3.cpp:1120`), so the
  warm hooks (`qwen3.cpp:813-864`) never prime the paged-KV shadows, so
  every decode's `TryPagedAttentionDeviceDecode` shadow-misses and the
  `EnsurePagedKvTtnn` VT_CHECK throws on the strided `KvSlice` view
  (`tenstorrent_ops.cpp:1179`) — 868 of 896 decode calls — and the host
  oracle runs. The default TT arm in the CLI has never exercised the device
  host-free path; both A/B arms are host-PA arms.
- The opt-out arm is measurably NOT faster because of PA: under
  `VT_TT_HOST_FREE_DECODE=0` the identical probe shows the identical regime
  (896 attempts / 868 failures / 0 device RAC / 0 warm hooks). The
  failed-attempt-overhead candidate is REJECTED (see `## Owed`).
- Capture-armed (`VT_TT_DECODE_CAPTURE=1`): warm hooks run, PA device
  failures drop to 0, capture arms (1 captured size) — and the FIRST
  capture dies: `TT_FATAL mesh_workload.cpp:153 !is_capturing_trace`.
  `EmbedDeviceIdsInto`'s `ttnn::copy(dev_out, *s->device)`
  (`tenstorrent_ops.cpp:5705`) is capture-only — the eager step's
  `EmbedInto` never runs it — so its program is cold mid-capture. The copy
  is unchanged since `79ff8f310`; the R5-era 27.1 tok/s single-request arm
  predates the QWEN35-wave edits to `tenstorrent_ops.cpp` (#1486
  cache-lifetime conversions, GDN staging), one of which dropped whatever
  accidentally warmed that program. The archaeology is not owed; the fix
  makes the invariant explicit.
- **The inversion is model-specific, not backend-wide (Mistral-7B A/B,
  same night, same recipe: 3 order-alternated pairs, `--repeat 5`,
  discard run 1, warm medians).** Mistral-7B-v0.3 on this P150: default
  host-free eager 11.51 tok/s vs opt-out 5.91 — the DEFAULT wins ~1.95x,
  the opposite of Qwen3-0.6B's 1.38x the other way. Against the R5-era
  record the default barely moved (12.2-13.8 then) while the opt-out more
  than doubled (2.35 then). The owed mechanism question therefore narrows:
  whatever makes the hybrid arm faster applies to Qwen3-0.6B and not to
  Mistral-7B; a single backend-wide explanation is ruled out by
  measurement.
- **The attempt-overhead mechanism is REJECTED on both models; the
  mechanism lives in the arms' successful paths (regime probe, 32 tokens,
  `VT_TT_TRACE_DEBUG=1`).** Per decode step the two arms' FAILED device
  work is identical on both models — Qwen3-0.6B: 896/896 PA attempts fail
  on both arms, 896 device-RAC attempts on the default arm with 0
  successes; Mistral-7B: 1024/1024 PA failures on both arms, 1024
  default-arm RAC attempts with 0 successes, 0 warm hooks anywhere. The
  default arm therefore does strictly MORE device work than the opt-out
  on both models, yet loses on 0.6B and wins ~2x on 7B. What differs is
  the work that SUCCEEDS: the host-free arm's device-resident R1 ops
  (RmsNorm/RoPE) succeed on both models and scale with hidden size
  (Mistral 4096 vs Qwen3 1024), while both arms run the same host PA
  oracle and host RAC fallback. Remaining attribution — how much of each
  arm's wall clock is the device R1 ops vs the host fallbacks — needs
  per-op timing and is future work, not this wave.

**Wave executed and green (2026-09-01, this branch, P150, all under one
`flock $HOME/gpu.lock`).** Both capture fatals are fixed and the
capture-armed arm is the fastest measured arm at tip.

- Fix 1 (embed, `qwen3.cpp` warm branch): the capture step now runs the
  exact captured embed segment once OUTSIDE the scope before opening it —
  the `_dummy_run` mirror. `EmbedDeviceIdsInto`'s hold semantics make the
  second run safe (the hold replaces; nothing reads the dummy output).
  This clears the first fatal (cold program mid-capture,
  `mesh_workload.cpp:153`).
- Fix 2 (fresh-slot zero, `tenstorrent_ops.cpp`
  `MemsetDeviceIfCapture`): `res.Zero` at the top of the captured layer
  region left the fresh slot host-only — the no-shadow arm refused,
  `MemsetDeviceFill` refuses under capture — so layer-0
  `EnsureDevice2D(*residual)` restaged from the recycled slot's stale
  persistent `[1,1024]` buffer: an enqueue_write, fatal at
  `fd_mesh_command_queue.cpp:760`. The no-shadow arm now serves the zero
  ON-DEVICE, capture-only: in-place into the slot's persistent buffer
  when the `[1, bytes/2]` geometry matches (stable device address, so the
  captured zero-copy replays against the same buffer), else a fresh
  `ttnn::empty` installed as persistent (W5). The zero tensor and the
  copy program are already warm: the cold step's `EnsureDevice2D` restage
  primes the zero at the exact spec (`ZeroCachePrime`) and the eager copy
  lane warms the program. bf16-only, the W7 reservation arm's polarity.
  CAPTURE-ONLY is load-bearing: the first attempt served eager fresh-slot
  zeros too, guessed bf16 from a byte size, and the f32 KV masters share
  those pool blocks — `EnsureHost` then aborted on a `[1,16777216]` bf16
  shadow against a `[256,32,8,128]` f32 request. An eager fresh-slot
  zero keeps the host fallback; bytes do not name a dtype.
- Focused gate (capture-armed 80-token CLI): mechanics GREEN — rc 0, 0
  fatals, 78 replays — but the answer was deterministic gibberish, not
  coherent. The "coherent" claim first recorded here was a measurement
  error (the gate's word-count criterion counted tt-metal log lines);
  corrected 2026-09-01 after the fresh reviewer's rerun flagged it, and
  root-caused as [#2461](https://github.com/mudler/vllm.cpp/issues/2461)
  (cold-step defect, pre-existing, replay-exact — only step 1 diverges).
  What survives of the original audit: `CaptureDecodePosAdvance` records
  inside the trace with no fatal, and replays are token-exact given their
  history (golden battery prompts 1-15 exact); no warm needed.
- Same-binary A/B/C (Qwen3-0.6B, 3 order-alternated triples,
  `--repeat 5`, discard run 1, warm medians): default 12.95 / opt-out
  17.68 / CAPTURED 27.57 tok/s. Captured replay is 2.13x the default and
  1.56x the opt-out; the R5-era 27.1 reproduces at tip. Every capture
  leg: replays=474, 0 fatals, rc 0, repeat-5 same-prompt multi-generate
  safe. CORRECTNESS CAVEAT (2026-09-01, #2461): the capture legs' output
  text was the deterministic cold-step corruption, so these ratios stand
  as measurements of the replay MACHINERY's speed only — not a capability
  verdict. Re-measuring a token-clean captured arm is part of #2461's
  repair gate.
- The verdict replicates at Qwen3-4B (same recipe, 2 triples): default
  9.25 / opt-out 8.86 / CAPTURED 13.90 tok/s — 1.50x / 1.57x, replays=474
  and 0 fatals on every leg. Captured replay dominates at both sizes.
  The default-vs-opt-out inversion narrows with hidden size exactly as
  the successful-path attribution predicts (0.6B opt-out wins 1.37x;
  4B default wins 1.04x; Mistral-7B default wins 1.95x) — but with the
  captured arm measured, the inversion no longer decides the flip: the
  captured arm beats BOTH eager arms at every measured size.
- Default arm untouched by the fixes: golden re-run 16/16, 125/125 PASS
  (the embed dummy also runs in the default arm's inert-scope warm step;
  no regression, same 0.375-nat pair).
- [#1625](https://github.com/mudler/vllm.cpp/issues/1625) boundary
  sharpened: same-prompt `--repeat 2` capture is clean (45 replays, one
  capture); the 16-prompt battery with capture armed spins in pure
  userspace after ONE capture (thread state R, stime=0, ~41 CPU-min;
  ptrace is unavailable on this host) — a cross-prompt KV-geometry
  re-warm under a live trace, tt-metal spin rather than the R5-era futex
  sleep. See `## Owed`.
- BLOCKED: the capture-armed golden battery RUNS (31/32 exact; the one
  failure is #2461's cold-step token) and the capture-arm near-tie pair
  stays blocked on #1625 (the `VT_DUMP_IDS` dump is 16 different prompts
  and spins). The wave's gate line "capture completes (no TT_FATAL)" is
  MET; "answer coherent vs the default arm under the near-tie rules" is
  NOT met — #2461 owns it, and with it the capture-default flip decision
  (blocked on #2461 + #1625).

**Wave scope (spec-first, one PR per the recorded row preference).** The R4
gate line "capture completes (no TT_FATAL)" is still unmet at tip; meet it.
On the capture step (`s.warm`), warm every capture-only segment before
`GraphCaptureScope` — the `_dummy_run` mirror: run `EmbedDeviceIdsInto`
once OUTSIDE the scope before the captured call, and audit the remaining
capture-only calls (`CaptureDecodePosAdvance`, captured RAC copies) the
same way, iterating until the capture-armed CLI completes 80 tokens with
replays > 0. Focused gate: capture-armed 80-token CLI run — no TT_FATAL,
`[Qwen3DenseDecodeGraph]` replay count > 0, answer coherent vs the default
arm under the near-tie rules. Then the real measurement: same-binary A/B,
captured replay vs hybrid opt-out. R5-era measured the captured arm at
27.1 tok/s against a 5.34 opt-out; the opt-out is now 17.8-18.6, so the
bar is "beat ~18.5". Only then do #1625 (captured multi-request hang) and
the capture-default flip decision come back within reach.
- **#2461 REPAIRED in this change (2026-09-01): the cold-step defect was a
  stride-fabricated KV shadow descriptor, not driver attn metadata.**
  `WarmPagedKvShadow` (`src/vt/tenstorrent/tenstorrent_ops.cpp:6278`)
  described the driver's flash-KV unbind(1) slice — a rank-4 view of the
  combined `[nb,2,bs,nkv,d]` store whose true block stride is
  `2*bs*nkv*d` (`dense_attn_block.h` `KvSlice`; observed shape
  `[256,32,8,128]`, stride `[65536,1024,128,1]`, `IsContiguous()==false`)
  — as `Tensor::Contiguous`, so `NhdToTtnnLayoutPrefix`
  (`tenstorrent_ops.cpp:834`) indexed the shadow prefix upload with dense
  strides: block `b` read `b*bs*nkv*d` elements from the view base, one
  slab early inside the combined buffer. For every physical block >= 1
  the K shadow received the previous block's V slab (zeros at prefill
  positions) and the V shadow the next block's K slab, so the cold step
  attended over zeros. Block 0 was accidentally safe because offset 0 is
  each view's own base; leg A never stages a shadow; and replays replayed
  the same corrupted shadow self-consistently — hence leg A coherence,
  token-exact replays (battery prompts 1-15), and the "consistent after
  step 1" shape. The fix stages the descriptor with the view's real
  strides and threads an `accept_unbind_view` flag (default false) through
  `EnsurePagedKvTtnn` → `NhdToTtnnLayoutPrefix`, admitting the view only on
  the warm/capture staging path; every existing caller keeps strict
  contiguity and the eager path is untouched.
  Red→green at this HEAD (Qwen3-0.6B, logs `$HOME/hf-repair-*`): RED —
  capture arm cold argmax 48755 (top-2 gap 1.69, word salad), default arm
  3364 and coherent, capture-armed golden battery 1 failed of 32
  assertions at prompt[0] tok=1. GREEN — capture arm cold argmax 3364
  (top-2 gap 1.125; default 3364, gap 1.375), text coherent;
  capture-armed battery 1 failed of 37 assertions, prompt[0] now exact and
  the single failure moved to prompt[1] tok=1 (engine 374 vs committed
  anchor 572 — the documented residual); default golden battery 125/125;
  `test_tenstorrent_backend` rc 0. Per-step adjudication
  (`VT_TT_DUMP_KV`, 12-token CLI pair, same binary): prefill and the cold
  step argmax-exact; the first divergence is step 2, capture top-2 gap
  0.375 nats — inside the near-tie band (`qwen3-neartie-gap.py`, ≤0.5) —
  after which both arms greedy-decode their own coherent prefixes.
  Token-clean A/B/C re-measurement (`hf-gate3.sh`, same binary, 3
  order-alternated triples, `--repeat 5`, cold repeat discarded, warm
  medians): default 12.90 / opt-out 17.80 / CAPTURED 27.47 tok/s — 2.13x
  the default, 1.54x the opt-out; replays=474 and 0 fatals on every leg,
  and every capture leg's output verified coherent this time. The captured
  replay ratios are now a capability verdict, not only a machinery
  measurement; the CORRECTNESS CAVEAT on the earlier 27.57 figure is
  superseded by this re-measurement.
  RESIDUAL, owned separately: the capture-armed 16-prompt battery still
  fails exactly one assertion, now at prompt[1] tok=1 — cross-request
  persistent shadow state, a different defect from the slab swap (prompt[1]
  passes in isolation). The battery's fatal REQUIRE stops the case at the
  first drift, so this run proves prompt[0] exact and the move of the
  failure cell; the scratch clone's verification covered prompts 1-15.
  Filed as [#2469](https://github.com/mudler/vllm.cpp/issues/2469); REPAIR
  LANDED in this change (2026-09-01) — the `## Owed` #2469 entry and the
  dated `## Now` repair block below carry the evidence.

**#2469 repair landed (2026-09-01, this branch, P150, all runs under
`flock $HOME/gpu.lock`).** The filed cross-request residual — a recycled
same-size slot replaying the previous request's final decode position — is
fixed at its three consumed surfaces.

- Root cause: `DecodePosCache()[num_reqs].cur_pos` is a persistent device
  tensor aliased into `RacIdxEntry.update_idxs` and `PaMetaEntry.cur_pos`
  and only `WarmDecodePos`'s seed branch re-seeds it; under capture the
  replay regime early-returns on `captured()` and `WarmPaMeta`'s
  `r2_steady` is process-global, so a recycled size slot kept the dead
  request's final position: PA attended the dead request's decode KV rows,
  RAC wrote at the stale virtual position. The investigation session's
  tokenizer decode showed the previous prompt's tokens (" Paris"/" France"/"
  is") from prompt[1] tok=1; the kept red artifacts document the failure as
  anchor drift with ids 374 (cap.out), 11 and 13 (diag/boundary).
- The repair: `Qwen3DenseDecodeGraph` size slots carry `expected_cur_pos`
  (seeded `seq_lens-1` on seeding steps, +1 per completed replay/capture
  launch) and `Step` gates `WarmDecodePos`'s replay regime on the
  continuation predicate `seq_lens[0]-1 == expected_cur_pos`; a same-size
  request boundary (captured && !continuation) routes through the proven
  #1476 re-capture lane — Reset the trace, run this step eagerly against
  the freshly seeded position, re-capture next step. On the device side,
  `DecodePosEntry.host_val` mirrors what `cur_pos` holds and `WarmPaMeta`
  echoes it into `cp_host`, repairing a vacuous guard: the unconditional
  `e.cp_host = cpos` made `TryPagedAttentionDeviceDecode`'s
  `cp_host[0] == seq_lens[0]-1` check compare the host against itself, so
  it could never fire on a stale device tensor. `plus_one_scratch` is
  allocated once per entry and reused, so the boundary seed performs no
  fresh device allocation under a live trace (the allocator's
  corruption-under-trace warning).
- Seed-only was tried and REJECTED: with the boundary seed verified
  correct on device, the first post-boundary replay still drifted to a
  near-tie wrong token (battery ids 11 and 13 across builds vs 374
  unfixed; id 264 appears in no kept artifact), so the boundary re-capture
  lane is required, not optional.
- Green on the stripped build (this session's diagnostic probes — KV row
  checksums, cur_pos/page-table/id readbacks — were investigation
  instruments and do not ride in the fix, per session precedent):
  capture-armed golden battery 125/125 assertions, 16/16 prompts PASS,
  max gap 0.375 nats @ prompt[1] tok=1 — inside the ≤0.5 near-tie band
  this row already adjudicates — 0 forward-divergent, `kPagedAttention
  selections=1344` with 0 declines; default arm unchanged (125/125,
  16/16, same max gap); captured CLI coherent and byte-identical across
  three runs including one after a `tt-smi -r 0` reset, and the
  default-arm CLI is coherent with a first-token near-tie flip — no
  stale-prompt tokens anywhere.
- The near-tie residual at prompt[1] tok=1 stays inside the row's ≤0.5
  band adjudication (max gap 0.375 nats), not a strict-token failure; the
  cross-request replay of the previous prompt's tokens — the actual #2469
  defect — is gone.
- **Token-clean at Qwen3-4B ([#2566](https://github.com/mudler/vllm.cpp/issues/2566);
  `hf-abc4b.sh`, 2026-09-01, 2 order-alternated triples at the repaired
  head `081efabc7`, same recipe: `--repeat 5`, discard run 1, warm
  medians): default 9.25 / opt-out 8.89 / CAPTURED 13.84 tok/s — 1.50x the
  default, 1.56x the opt-out; replays=474 and 0 fatals on every leg, and
  the capture legs are coherent and deterministic (the pre-repair 4B
  capture legs were the #2461 cold-step gibberish). The CORRECTNESS CAVEAT
  on the earlier 13.90 machinery-only figure is superseded by this
  re-measurement; captured replay dominates both eager arms at 0.6B and 4B
  on token-clean evidence.
- **Token-clean at Mistral-7B-v0.3 ([#2566](https://github.com/mudler/vllm.cpp/issues/2566);
  `hf-mist-abc.sh`, 2026-09-01, 2 order-alternated triples at
  `081efabc7`, `--max-tokens 64 --repeat 5`, `tt-smi -r 0` first, warm
  medians): default 11.51 / opt-out 5.93 / CAPTURED 14.23 tok/s — 1.24x
  the default, 2.40x the opt-out; 314 replays, 0 fatals, rc 0 on both
  triples, and the capture arm's output is TOKEN-IDENTICAL to the default
  arm's in both triples (the A-vs-B text divergence is the known
  eager-kernel near-tie situation). This is the model with the sharpest
  default-vs-opt-out inversion (the default wins ~1.95x), and the captured
  arm beats both — captured replay's superiority over eager now holds at
  all three measured sizes on token-clean legs. The per-model record entry
  is in `.agents/benchmark-record.md`.

## Capture-default flip (#1625)

The captured decode arm becomes the DEFAULT on Tenstorrent Blackhole:
`support_static_graph_mode()` returns `HostFreeDecodeEnabled() &&
DecodeCaptureEnabled()`, where `DecodeCaptureEnabled()` mirrors
`HostFreeDecodeEnabled()`'s polarity (`VT_TT_DECODE_CAPTURE` unset or any
value except "0" arms capture; `VT_TT_DECODE_CAPTURE=0` opts out and restores
today's eager host-free arm). The platform comment that declined this flip —
"the captured arm hangs deterministically on the FIRST multi-request run" —
names #1625, and that hang is root-caused and repaired on this row: the
short-chunk device KV push clobber (#2669, this branch) was the defect, the
captured multi-request battery is 16/16 green and hang-free across the
focused, recap8, and full-battery lanes, and the flip it held back is now
unblocked. The captured arm is also the fastest measured arm (28.61 tok/s
warm median against 12.21 for the eager host-free default and 15.61 for the
host opt-out, same-binary order-alternated triples, #2566 recipe), so the
flip ships the payoff rather than a risk.

### Scope

- `src/vt/tenstorrent/tenstorrent_device.h`: add `DecodeCaptureEnabled()`
  beside `HostFreeDecodeEnabled()`, same one-line parse.
- `src/vllm/platforms/tenstorrent.cpp`: rewrite the R5-flip comment block to
  record WHY capture is now default (the #2669 root cause and the measured
  payoff) and how to opt out; change the one-line conjunct.
- `tests/parity/test_qwen3_paged_engine.cpp`: the `_capture` golden selection
  currently keys on `std::getenv("VT_TT_DECODE_CAPTURE") != nullptr`
  (presence, not value). It must key on the SAME parsed polarity as the
  platform, so `VT_TT_DECODE_CAPTURE=0` selects the eager pair in the test
  exactly as it selects the eager path in the engine. A drift here produces a
  test that passes while the engine runs a different arm than the goldens
  adjudicate.
- Qwen3-4B captured bring-up (AMENDED 2026-09-03 — outcome in
  `## Capture-default flip (#1625)` → `### Bring-up amendment`): the pair was
  dumped, teacher-forced, and CANNOT be committed — both TT arms land beyond
  the near-tie band. Owned by #2811.
- `docs/FEATURES.md` TT host-free row: "Capture opt-in only (#1625 hang)"
  becomes "capture DEFAULT since #<PR> (`VT_TT_DECODE_CAPTURE=0` opts out)",
  with the payoff figure.
- Benchmark publication: `docs/BENCHMARKS.md` has no Tenstorrent entry. The
  flip owns one `docs/benchmarks/<benchmark-id>.md` detail file plus its
  index row, publishing the #2566-recipe rate figure RE-TAKEN ON THE FLIP
  TREE (capture is the default there, so the published number must be the
  shipped default, not the opt-in arm): capture/default leg, opt-out leg,
  order-alternated triples, `--repeat 5`, leg 1 discarded, warm medians,
  flock + tt-smi reset per batch.
- Closes #1625 (hang root-caused to #2669; flip landed; multi-request
  captured battery hang-free).

### Not in scope

- The host opt-out inversion residual (eager host-free 12.21 vs host opt-out
  15.61 tok/s) — pre-existing, stays open on its own lane.
- #2670 (reader-side `device_current` trust) and #2671 (dump verdict mark) —
  remain `## Owed` on this row.
- Async readback (#1627) — unchanged.

### Tests and gates (red-first)

1. RED-FIRST (no card needed): a default-polarity assertion — with NO env
   set, the engine arms capture and the test selects `_capture` goldens; with
   `VT_TT_DECODE_CAPTURE=0`, both stay eager. On the pre-flip tree this reds
   (capture requires env presence); on the flip tree it greens.
2. The focused #2669 boundary gate and the captured 0.6B battery run with NO
   capture env (they now arm capture by default): 2/2 and 16/16 against the
   committed capture pair.
3. `VT_TT_DECODE_CAPTURE=0` runs the eager arm on the flipped tree: 0.6B
   eager battery 16/16 anchor-exact (the eager pair keeps gating the opt-out
   arm).
4. 4B battery with NO env (captured default): 16/16 against the
   brought-up pair (or byte-equality evidence).
5. recap8 lane with NO env: 16/16 (recapture cadence survives the flip).
6. `test_tenstorrent_backend` full suite; site suite (docs changed);
   staged preflight; fresh review with mutations (threshold-style: revert
   the platform conjunct and watch gate 1/2 red).

### Stop conditions

- If the captured 4B arm fatals or hangs on the P150 (the 0.6B capture
  fatals were fixed in an earlier wave, but 4B has NEVER run captured), the
  flip scopes down to 0.6B-only: the 4B case pins `VT_TT_DECODE_CAPTURE=0`
  explicitly with a comment naming the Owed 4B-capture issue, and the row
  records the fatal as owed evidence. A default the test cannot reach is not
  shippable, so the flip does NOT land with 4B captured-but-ungated.
- If the flip-tree benchmark shows the captured default regressed below the
  eager host-free default (it should not; same code path as the 28.61
  measurement), stop and re-root-cause before landing.

### Decisions recorded

- One PR for spec and implementation (user: "prepare the next PR",
  2026-09-03; matches the recorded pattern for the other TT rows).
- 4B captured pair is IN this PR (a default the tests never exercise is the
  nothing-lands-dead smell; scoping it out requires the stop-condition
  evidence above).
- The published benchmark figure is re-taken on the flip tree.

### Bring-up amendment (2026-09-03)

The 4B bring-up the scope bullet promised produced findings the spec did not
anticipate. Recorded here because the code and Git history do not carry the
decisions.

**Finding 1 — the 4B TT forward diverges beyond the near-tie band on BOTH
arms.** Both arms are deterministic (capture dumps byte-identical across runs,
sha256 `7e77e6ea…`; eager `8325ff10…`) and the two fatal bugs on the 4B
capture path are FIXED (commit `1872aeb16`: D2D copy and capture memset took
`BufferSlot::bytes` — the #1922 best-fit lend's host-block CAPACITY — as the
content contract; on 4B the 5120-B `s.hidden` shadow sat in a 10240-B block
and the D2D lane declined, falling through to a host read mid-capture
(TT_FATAL, `fd_mesh_command_queue.cpp:807`), and the memset derived its zero
geometry from capacity and staged a [1,5120] zero no warm step primed
(zero-cache miss, `tenstorrent_ops.cpp:255`). 0.6B survived on its size-class
bins.) But against `transformers`-teacher-forced gaps on each arm's OWN
prefix, 5 of 16 prompts land beyond the 500-mnat band — capture worst 1000
mnats @ (11,3), eager worst 2000 @ (5,6); first-divergence margins 0.625-2.0
nats, which is a real forward difference and not a bf16 tie (0.6B TT worst
500; the committed CUDA 4B pair's worst 250). A pair that fails its own gate
cannot be committed. DECISION (user, 2026-09-03): scope 4B down per the stop
condition's pattern — the 4B dense gate skips loudly on TT naming #2811,
`VT_DUMP_IDS=1` still bootstraps a dump for the bring-up, and the finding is
owed. Hypothesis to falsify on #2811's lane: diffuse bf16 accumulation drift
at H=2560/36L (the divergence is concentrated: most prompts are fully
vLLM-endorsed, gap 0 on every cell).

**Finding 2 — the flip's blast radius was every TT decode-graph driver, and
one of them crashes.** `support_static_graph_mode()` is consulted by ten
driver sites beyond Qwen3-dense (Qwen3.5-GDN ×2, Qwen3.5-dense, Qwen3-MoE
driver + registry, GLM4.5-MoE-lite registry, DeepSeek-V2 driver + registry,
Qwen3-DFlash, Voxtral). The flip as first committed armed capture for ALL of
them by default; the Qwen3.5-0.8B gate battery TT_FATALs mid-capture
("Writes are not supported during trace capture",
`fd_mesh_command_queue.cpp:760`) — a product-default crash for a model that
worked pre-flip. Mistral-7B (which rides the Qwen3-dense machinery) completes
but its captured sequence drifts from its committed EAGER pair (anchor drift
p[1] t=8, `engine=1924 anchor=1988`) — pair debt, not a crash. SHAPE: the
platform gains `static_graph_requires_opt_in()` (base false; TT returns
`!DecodeCaptureRequested()`), every driver conjuncts the seam — the ten
non-Qwen3-dense sites call the no-arg overload — and the Qwen3-dense driver
calls the
architecture-aware overload, because its ONE graph class also serves the
Mistral, Llama and InternLM2 registries, and the TT override carves default
capture to exactly `Qwen3ForCausalLM` (DECISION, user, 2026-09-03: carve by
architecture — an ungated default arm for three registries is the failure the
scope-down exists to prevent). Those three keep the pre-flip explicit opt-in
until their captured arms are brought up (#2812). The Qwen3.5 and Mistral
gate harnesses mirror the engine arm (`DecodeCaptureEnabled() &&
DecodeCaptureRequested()`); a hardcoded eager name would adjudicate a
captured run against eager goldens.

**Gates this amendment adds:** Qwen3.5-0.8B ambient (no env) greens against
its committed ambient pair (pre-amendment tree: TT_FATAL red,
`flip-q35-prefix.out`); Qwen3.5 with `VT_TT_DECODE_CAPTURE=1` loud-skips;
Mistral ambient greens against its committed eager pair — the architecture
carve restores the pre-flip eager default (pre-carve red: captured-vs-eager
anchor drift, `flip-mist-prefix.out`); Mistral with `VT_TT_DECODE_CAPTURE=1`
loud-skips; 4B dense loud-skips (#2811); 0.6B both
legs and the backend suite stay green. `flip-{q35-ambient,q35-optin,mist-skip,4b-skip,
06b-ambient,06b-eager,backend-scope}.out` hold the green side.
