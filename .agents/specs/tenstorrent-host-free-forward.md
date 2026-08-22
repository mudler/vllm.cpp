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
- **`test_tenstorrent_backend` exits 139 after a fully green doctest summary**
  (23/23 cases, 831/831 assertions): static `std::optional<ttnn::Tensor>` cache
  fields are destroyed after the UMD device closes and `deallocate_impl` reaches
  a torn-down `GraphTracker`. Proven pre-existing at `origin/main` by an A/B
  stash/build/run during the [#1476](https://github.com/mudler/vllm.cpp/issues/1476)
  gate. Owned by
  [#1486](https://github.com/mudler/vllm.cpp/issues/1486).
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

The operator must still rerun the 80-token no-hang gate and
`test_qwen3_paged_engine` on a Blackhole P150. An implementer run is an
input, not a gate result.

## Now

`ACTIVE`. R1-R3b and the R2 on-device `cur_pos` / `update_idxs` advance are
implemented on this branch, env-gated by `VT_TT_HOST_FREE_DECODE`.

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
