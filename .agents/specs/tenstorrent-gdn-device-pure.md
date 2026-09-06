# BACKEND-TENSTORRENT-GDN-DEVICE-PURE — device-resident GDN decode for trace capture

Row: `BACKEND-TENSTORRENT-GDN-DEVICE-PURE` · Issue: #2907 · Owed from #2812
(`.agents/specs/tenstorrent-host-free-forward.md` `## Owed`). Lifecycle:
`ACTIVE` at the spec commit; implementation follows in this same PR (one-PR
shape is the recorded row-claim answer, 2026-09-04).

## Scope

Decode path only (T=1) inside the `Qwen3_5DenseDecodeGraph` Step. Make the
decode-side GDN ops device-resident so a tt-metal trace capture admits them,
unblocking the Qwen3.5-0.8B captured arm — the last capture-blocked family
with a committed eager pair.

IN:
- `CausalConv1dUpdateKernel` (`src/vt/tenstorrent/tenstorrent_ops.cpp:5013`)
  and `GdnDecodeKernel` (`tenstorrent_ops.cpp:5225`): remove the per-call
  host orchestration — `EnsureHost` on x/q/k/v/g/beta, `ReadIdxHost` on the
  state index, per-call host vector builds, per-call `UploadTensor`
  (`:5225` region, five-plus sites) — and replace it with device-resident
  inputs, device-side indexed state update, and no host reads or writes
  inside the captured region.
- Whatever must restage per step moves under the recapture cadence's
  re-priming instead of inside the captured span.
- The q35 harness (tests/parity/test_qwen35_paged_engine.cpp, pair selection
  ~:233) grows the captured-arm selection and the committed capture pair;
  the #2812 loud-skip cells run for real.
- Admission of the Qwen3.5 architecture to `DecodeCaptureDefaultArch`
  (tenstorrent_device.h:85 region) once the pair gate passes.

OUT: prefill capture (`kGdnPrefill` stays as landed), MoE decode graphs
(`Qwen3_5DecodeGraph`), recapture-policy changes, Llama/InternLM2 pairs,
performance floors (see Stop conditions).

Prerequisites (landed in the base): #2812's two staging-class fixes
(`ebce75456` gemma slot + weight-view shadows) — the fatal that remains is
structural, repro 2/2 (`pairs-q35-run1.out`, `pairs-q35-repro1.out`):
"Writes are not supported during trace capture"
(`fd_mesh_command_queue.cpp:760`).

## Our baseline

| Anchor | State |
|---|---|
| `src/vt/tenstorrent/tenstorrent_ops.cpp:5612` `EnsureConvStateTransposed` | Decode's transposed conv shadow ([sl+1, slots*C] f32). Before this row its slow path `EnsureHost`ed (a download) before its refusal CHECK — the readback that killed the captured arm mid-capture. |
| `src/vt/tenstorrent/tenstorrent_ops.cpp:4840` `EnsureGdnCacheDevice` | Prefill's cache-geometry view of the same buffer. Replaces the slot's device tensor, which is what strands a baked replay pointer. |
| `src/vt/tenstorrent/tenstorrent_ops.cpp:2042`, `:2079` | `kCastBf16` / `kCastF32`. Baseline: unconditional host path (`EnsureHost` + scalar cast), which inside a captured graph is either a no-op (host-current bytes) or the fatal readback this row refuses cleanly. |
| `src/vt/tenstorrent/tenstorrent_device.h:85` | `DecodeCaptureDefaultArch` — Qwen3-dense and Mistral admitted; Qwen3.5 absent, so its captured arm was opt-in-only and its opt-in cells skipped loudly (#2812). |
| `src/vllm/model_executor/models/qwen3.cpp:912-914` | The qwen3 dense driver's `tt_boundary` recapture port (#2469) — the shape the GDN dense Step lacked. |
| `tests/parity/test_qwen35_paged_engine.cpp` | The 0.8B gate: pair keyed on the committed goldens; before this row the opt-in cells could not run captured at all. |

## Upstream chain

- vLLM is the behavior mirror for GDN semantics; the TT decode kernel
  already ports it and the committed eager pair adjudicates it. Ported
  sites are cited in the GDN parent row's records; this row changes
  orchestration, not recurrence math — any math edit must cite the upstream
  `file:line` it mirrors and re-run the backend op-level oracle suite
  (`test_tenstorrent_backend.cpp:1749-3340`, CPU f32 arm).
- Pair oracle: `transformers` via `scripts/qwen3-neartie-gap-transformers.py`
  (secondary oracle, `~/Sources/tt/tt-metal/python_env/bin/python`), the
  Mistral-precedent treatment; vLLM has no TT backend.

## Port map

| Upstream | Local | Note |
|---|---|---|
| qwen3 dense driver boundary port, `src/vllm/model_executor/models/qwen3.cpp:912-914` (`tt_boundary` = TT && !seq_continuation, widening the recapture predicate) | `src/vllm/model_executor/models/qwen3_5.cpp:11626-11629`, dense Step | Intra-repository port of our own #2469 shape: a prefill join must end the captured run's continuation so the next decode re-captures instead of replaying stale baked pointers. |
| — (no upstream analogue; tt-metal has no shadow-serveability query) | `src/vt/tenstorrent/tenstorrent_ops.cpp:7352` `ConvShadowServeable` | Read-only predicate over the slot flags; lets the Step distinguish "shadow servable" from "prefill replaced it" without touching state. Recorded in the porting inventory as a scratch-free local design. |
| — (local; enforces the tt-metal trace rule "no reads during capture") | `src/vt/tenstorrent/tenstorrent_ops.cpp:2042`, `:2079` | `kCastBf16`/`kCastF32`: device-shadow fast path; fallback to the host path only while the bytes are host-current, else a named refusal instead of a raw tt-metal abort. |
| — (local; orchestration only, no recurrence math) | `src/vllm/model_executor/models/qwen3_5_dense.cpp:186` + both `Impl` ctor gate sites | `static_graph_requires_opt_in(config.architectures)` — the arch-scoped overload, so admission travels with the checkpoint's architectures field. |

## Tests to port

vLLM has no TT backend and no captured-arm test to port; the oracle chain is
the Mistral precedent's. The gate family is local and asserts the same
invariants the Qwen3/Mistral gates assert: two-run byte-identity across a
card reset, teacher-forced token/gap pairs against the `transformers`
reference (the secondary oracle), ambient-captured vs opt-out-eager
adjudication, and load-bearing mutations (pair tamper, admission deletion).
The 16-prompt battery shape and the near-tie 500-mnat band are the committed
q35 goldens' existing conventions.

## Design (shape; the implementer refines with traced evidence)

1. Inputs resident: the Step's GDN operands must arrive as device tensors
   from the preceding captured ops, not as host tensors the kernel stages
   per call. Where the engine hands a host tensor, the graph's producer op
   commits device-side and the kernel consumes the shadow.
2. Indexed state without host: `ReadIdxHost` addresses the state slab by a
   host-side slot index. Decode slots are stable across a sequence's steps,
   so the captured graph can bake the addressing and let the recapture
   cadence re-prime on slot change; the device-side alternative is
   indirection through `kGdnStateGather`/`kGdnStateScatter` on a device
   index tensor. Measure the recapture cost before choosing; if per-step
   recapture destroys the win, indirection wins by default.
3. Conv-state coherence: the conv update keeps its two views
   (hidden-state view, ring view) coherent on device — see the conv-state
   two-views record in session memory before touching the layout.
4. Outputs commit device-side; token readback stays outside the captured
   span, as the dense path already does.
5. Found root cause (traced, 2026-09-05): the GDN `conv_state` buffer is
   DUAL-ROLE — decode keeps a transposed shadow ([sl+1, slots*C] f32,
   `EnsureConvStateTransposed`), prefill gathers it in cache geometry
   (`EnsureGdnCacheDevice`). A prefill-bearing step (new request joining)
   re-uploads cache geometry, clears the slot's `conv_transposed` flag and
   REPLACES the slot's device tensor; the next decode replay then reads its
   baked stale pointer (uniform tok3 divergence, persistent sentinels), and
   with the boundary port that decode becomes a capture whose
   `EnsureConvStateTransposed` slow path DOWNLOADS during capture
   ("Reads are not supported during trace capture"). Pointer replacement is
   non-self-healing, which is why the fatal persisted. Fix, four parts:
   (a) the qwen3 `tt_boundary` recapture port (qwen3.cpp:903-921) into the
   dense Step; (b) `ConvShadowServeable` — a read-only predicate over the
   slot flags that gates shadow serving and, when false, widens the Step's
   recapture lane so an eager step rebuilds the shadow and the next capture
   bakes fresh pointers (converges); (c) CHECK-before-download reorder in
   both slow paths so a capture-time call refuses instead of downloading;
   (d) `Qwen3_5ForConditionalGeneration` joins `DecodeCaptureDefaultArch`.
   No recurrence math changed; the backend op-level oracle suite is
   re-run by the battery.

## Tests (red-first)

- RED now: the q35 opt-in battery's capture-blocked cells skip loudly
  (#2812 skip at the `test_qwen35_paged_engine.cpp:504` case); flipping them
  to run fatals deterministically (`fd_mesh_command_queue.cpp:760`). The
  wave's first green is those cells running captured.
- Byte-identity: captured dump ×2 with a card reset between
  (`VT_DUMP_IDS=1`), byte-equal — the Mistral precedent.
- Pair: teacher-forced via the transformers oracle; every cell inside the
  committed eager pair's band (or the 500-mnat Mistral band if the eager
  band is looser); worst cell recorded in the pair commit message.
- Admission: Qwen3.5 joins `DecodeCaptureDefaultArch`; ambient adjudicates
  the CAPTURED arm; `VT_TT_DECODE_CAPTURE=0` keeps the eager arm against
  the eager pair. Harness selection keys on `DecodeCaptureEnabled()` (the
  q3/mistral pattern), never on the env being set.
- Mutations: tamper one capture-pair cell → ambient reds at the anchor;
  delete the arch line → ambient reds (the #2812 mutation pattern).

## Dependencies

| Dependency | State |
|---|---|
| `BACKEND-TENSTORRENT-GDN` op chain (kernels + CPU f32 oracle suite) | DONE — this row changes orchestration, not recurrence math |
| `BACKEND-TENSTORRENT-QWEN35` wiring row (allow-list entry, e2e pair, W1-W4 levers) | DONE; its captured-arm blocker is this row's subject (#2812 → #2907) |
| `BACKEND-TENSTORRENT-HOST-FREE-FORWARD` (Mistral captured pair + admission precedent, #2566) | DONE — the pair-refresh and mutation pattern copied here |
| pinned tt-metal trace capture (W0 skeleton, recapture cadence) | pinned per `.agents/upstream-sync.md`; RC cadence is a local stability measure |

## Work breakdown

| W | Content |
|---|---|
| W0 | The spec, committed alone (`5b0d001fe`) before any implementation. |
| W1 | RED: the captured arm fatals on a host readback mid-capture; CausalConv1dUpdate + GdnDecode device-resident; focused suite green. |
| W2 | GdnPostConv, RmsNormGated, AttnQkNormRopeGate device-pure; device-side cos_sin production; 2023/2023 focused green. |
| W3 | Root cause (dual-role shadow replacement at prefill joins) traced with an instrumented backtrace; the four-part fix; probe green (375 mnats, zero sentinels). |
| W4 | Pair refresh + commit (Gate 1-2), admission + ambient/opt-out adjudication (Gate 3), mutations (Gate 4), full battery (Gate 5), records + commit (Gate 6). |

## Gates

- Full battery on the head: 06b/q35/mistral ambient+opt-out, q35 opt-in with
  NO skips left (all cells run captured), 4B skip #2811 unchanged, backend
  suite 52/52; preflight rc=0; fresh review with mutation proof; the
  operator reruns the battery itself.
- Perf is report-only: captured-vs-eager tok/s on 0.8B re-measured on the
  landed tree (the 2026-09-04 eager wall is 0.177 tok/s). No ratio floor is
  claimed or gated.

## Evidence

Session logs under `/tmp/r1625-issues/` (`pairs-q35-*.out` fatal repros;
this row's legs prefixed `gdn-`). Committed goldens carry the model
snapshot revision in the pair manifest.

Pair refresh (2026-09-05, same model snapshot
2fc06364715b967f1860aea9cf38778875588b17): once the captured arm ran
end-to-end, BOTH pairs were re-derived from the live tree — the captured
pair (`*_tenstorrent_capture.npy`: 68 cells diverge from HF greedy, max
gap 375 mnats, zero sentinels) and the eager pair (`*_tenstorrent.npy`:
81 cells, max 375 mnats, 69/81 our-token-is-argmax). The refresh is a
justified numerical change, not gate-silencing: the pre-refresh pairs were
captured on the staging-per-call tree whose arm never completed a captured
run; the arms differ at 59/256 cells (near-ties on both paths), so a
shared stale pair could not adjudicate either arm.

## Risks

- Recapture-per-step cost: if slot changes force recapture every step, the
  captured arm loses to eager — the design's clause 2 decides on measurement.
- A third structural staging class may surface once the two kernels are
  device-pure (unknown unknowns behind the fatal).
- The conv-state two-views layout is fragile; a layout change touches the
  prefill path's state handoff even though prefill capture is out of scope.

## Stop conditions

- The captured arm still fatals after both kernels are device-pure: name the
  next blocker with `file:line`, record it, stop.
- Pair outside the band after two repair attempts: `NEEDS_DECISION`.
- Never declare a same-architecture performance ceiling; an apparent limit
  is an unresolved difference — keep the gap open and name the next
  traceable hypothesis.

## Owed

- Qwen3.5 MoE-family captured arms (out of scope here; `Qwen3_5DecodeGraph`).
- Llama/InternLM2 captured pairs: demoted behind #1627 multi-request
  evidence (2026-09-04 bench: Mistral capture = 1.00x at c=1/7B, 0.6B =
  2.16x; their payoff is contingent on concurrency).
- #1625's multi-request captured throughput claim (async lane) is untouched.

## Outcome (2026-09-05)

- Landed: `108865e7a` (the device-resident ops and the capture admission)
  plus `bcade48d6` (the ratchet-pin evidence tying the BACKEND count to the
  row), through PR #2956; the merge closed #2907 and #2812.
- Measured: captured and eager arms differ at 59/256 cells (near-ties on
  both paths); each arm against HF greedy stays inside the 500-mnat band
  with worst cell 375 mnats (captured 68 cells, eager 81 cells,
  69/81 our-token-is-argmax). Ambient (default captured) and opt-in
  captured legs both adjudicate 146/146 against the committed capture
  pair; the recapture cadence in the opt-in leg is
  `VT_TT_RECAPTURE_EVERY=4` — 0 and 8 both froze under prefill joins.
- Rejected: a blanket capture-time refusal in `kCastBf16`/`kCastF32`.
  The Qwen3-0.6B and Mistral captured arms legitimately cast
  host-current tensors mid-graph, so their ambient legs threw; the
  landed form serves the device shadow when one exists and otherwise
  falls back to the host path, refusing only when the bytes are not
  host-current (the readback capture forbids).
- Defaults: harness pair selection keys on `DecodeCaptureEnabled()`
  alone (the q3/mistral pattern); admission covers only the dense
  driver whose captured arm has a committed pair; the eager pair is
  re-derived rather than reused, because the prior pair predates any
  completed captured run and the arms differ at 59/256 cells.
- Perf stays report-only per the Gates section; no ratio floor claimed.
