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

- **`DecodePosCache` is keyed on bare `num_reqs`, with no engine, queue or device
  identity, and is never cleared.** Two engine instances in one process at the same
  padded batch size therefore share one `cur_pos` device tensor: the second engine's
  first `WarmDecodePos` finds the first engine's entry, returns early, and both
  `WarmRacIdx` / `WarmPaMeta` aliases bind to a buffer another engine is advancing.
  That is silently wrong rather than a refusal, and it is a candidate explanation for
  `test_qwen3_paged_engine` still timing out under the flag. Owned by
  [#1105](https://github.com/mudler/vllm.cpp/issues/1105).
- **`VT_TT_RECAPTURE_EVERY` lags `cur_pos` by one per recapture cycle.**
  `GraphCapturesCounter` is only ever `fetch_add`ed (`tenstorrent_ops.cpp:3225`, called
  once from `:3297`); nothing resets it, and `DestroyGraph` does not. So the
  recapture-triggered eager step re-seeds nothing and skips every `copy_to_device`,
  and the following capture step captures a `cur_pos` one behind. The eager PA
  consistency check at `:2410-2412` cannot see it: it compares `e.cp_host[0]` against
  the same `seq_lens` that rewrote `e.cp_host` at `:3982`, so it validates the host
  mirror against itself. Owned by [#1105](https://github.com/mudler/vllm.cpp/issues/1105).

The seven constraints above remain. A new batch size after the first
capture is now refused (`VT_CHECK` in `WarmDecodePos` / `WarmPaMeta` /
`WarmRacIdx`) rather than freezing `cur_pos`. The real fix is a
per-cache-entry seed / generation field and aliasing on every warm call,
not a process-global `GraphCapturesCounter`. Tracked on
[#1105](https://github.com/mudler/vllm.cpp/issues/1105).

The operator must still rerun the 80-token no-hang gate and
`test_qwen3_paged_engine` on a Blackhole P150. An implementer run is an
input, not a gate result.

## Now

`ACTIVE`. R1-R3b and the R2 on-device `cur_pos` / `update_idxs` advance are
implemented on this branch, env-gated by `VT_TT_HOST_FREE_DECODE`. A P150
run of Qwen3-0.6B "Hello" at 80 tokens completed 79 replays with no hang
and 22/22 argmax vs the per-step-copy baseline. Next: operator rerun of
that 80-token gate and `test_qwen3_paged_engine` on card. A new batch
size after the first capture now throws instead of emitting wrong tokens.
