# The 27B captured step decompose — the ranked cost table (2026-09-26, P150a)

**STATUS: measured; the table is complete and the dominant term is named with
its mechanism.** Row `BACKEND-TENSTORRENT`, spec
`.agents/specs/tenstorrent-27b-step-decompose.md`, branch
`row/TT-27B-STEP-DECOMPOSE`, issue `ISSUE-LOCAL-01M3F7GQ4MXQT72D05A12JD9EQ`.
W0 scope held: instruments and measurement only — no kernel, no capture-path,
no optimization landed.

## Verdict up front

The ~35 s/token is **not warmup bookkeeping, not the capture finalize, not
retention pressure, not JIT, and not kernel compute**. It is the **per-op
dispatch overhead of the S=1 forward**, paid twice per request on two
different paths for the same ~1.3 s of actual kernel work:

1. **64.4% — the eager cold body: 67.0 s/request (2 of 3 decode steps), host
   dispatch.** The dense captured arm's slot state collapses to `cold` at
   EVERY decode step, so two of the three steps run the full 64-layer forward
   EAGERLY, at ~33.5 s of HOST time each — while the device does **1.27 s** of
   work (the legD probe: gdn 20.3 ms, fa 18.6 ms per layer, 64 layers =
   1.27 s total) and is finished long before the host is. The forward is
   **1,037 tt-metal op entries** (legE's in-capture `[TT-OP]` census, identical
   across all four captures), so the eager path pays ~32 ms of host dispatch
   per op.
2. **29.9% — the trace-execution wait: 31.1 s/request (1 of 3 steps).** The
   one step that does capture replays the trace and returns; the sampler's
   blocking read then waits **31.1 s** for a trace whose kernels are the same
   ~1.3 s — ~29.8 s of trace-replay overhead, ~30 ms per replayed command
   over the same 1,037-entry trace.
3. **5.5% — the capture pass: 5.6-5.9 s/request.** cap_body (the record pass:
   ~2.6 s of host record cost, 41-46 ms/layer) + cap_end (~1.7 s, tt-metal's
   `end_trace_capture` finalize) — the capture ECONOMICS are cheap; the arm
   just never gets to use what it captures.
4. **<0.1% — the per-step TT refresh block** (`warm_calls=20`, warms+embed+
   replay-launch ≈ 2 ms) and the sampler's own read (~1.6 ms).

Genuine kernel compute is ~1.3 s per forward — **under 4% of the wall**. The
retention staircase does not step in this arm (per-request entry DRAM repeats
byte-flat; >5 GiB free at request 4), and no JIT fires after request 1.

### The mechanism of the collapse (the 64.4% term's cause)

`VT_DECODE_GRAPH_STATS` prints `4 total replays` for a 12-decode-step run:
every capture's own launch, never a served replay. The per-step phase lines
print `boundary=1` at every decode step: the #2469 continuation predicate
fails always, so the `tt_boundary` lane resets the captured graph before it
can serve. The dense driver's port of qwen3.cpp's R2 cur_pos bookkeeping
**missed the two post-replay increments** — qwen3.cpp:971 (`++s.expected_cur_pos;
// the replay's in-trace plus_one advanced cur_pos`) and qwen3.cpp:1118 (`++s.expected_cur_pos;
// the capture step's launch ran the trace once`) exist in the qwen3 driver;
the dense driver (`Qwen3_5DenseDecodeGraph::Step`) updates
`s.expected_cur_pos = pam.seq_lens[0] - 1` (qwen3_5.cpp:12400) and checks
`(seq_lens[0]-1) == s.expected_cur_pos` (qwen3_5.cpp:12395) but never
increments after `s.graph.Replay` — so the check is false at every
mid-request step, `s.graph.Reset()` fires (qwen3_5.cpp:12437), and the slot
goes back to cold. Per request the decode sequence is [cold 33.9 s → capture
5.7 s (+31.1 s wait) → cold 33.2 s] where it should be [cold → capture →
replay].

### The lever this names (the next perf row)

Two moves, in order:

1. **Complete the #2469 port** (two `++s.expected_cur_pos` lines in the dense
   driver's replay and capture-tail branches). This is a capture-path change —
   out of THIS row's scope by its non-goals — and it is filed as the follow-up
   issue. Expected value: the third step becomes a replay launch (~0.3 ms) +
   the 31.1 s trace wait — a small direct gain (33.2 → 31.1 s, ~-6%), but it is
   the precondition for every captured-arm lever to be measurable at all.
2. **The per-op dispatch overhead itself.** With the port complete, the wall
   is ~31.1 s of trace-replay overhead + ~33.5 s × (steps the collapse no
   longer forces) — and the next row targets the ~30 ms per op: the eager host
   dispatch (~33.5 s over the 1,037-entry op census, legE) and the
   trace-replay's per-command cost (31.1 s for the same op set). That is an
   op-count/fusion lever (ours — 572 of the 1,037 entries are matmul-class)
   and a dispatch-cost lever (tt-metal's lane) — with the kernels at 1.3 s,
   a fused-ops row has a ~25× headroom on paper.

## The instrument

`VT_TT_STEP_PHASES` (read-only, zero-cost when unset, documented in
`docs/ENVIRONMENT.md`):

- One `[TT-STEP-PHASE] step=...` line per `Qwen3_5DenseDecodeGraph::Step`:
  kind (cold/capture/replay/eager-fallback), the phase brackets (warms, embed,
  pregrow, cap_begin/cap_body/cap_end, pin, replay launch, body), the step
  wall, `gap_prev` (the runner/sampler window to the previous return), the
  per-step warm-call count, and `dram_free` (the per-step retention read-out).
  Emitted at every return the captured arm takes
  (`src/vllm/model_executor/models/qwen3_5.cpp`, the `StepPhaseEmit` sites in
  `Qwen3_5DenseDecodeGraph::Step`).
- Seam-side `[TT-STEP-PHASE] sync` lines: the blocking-read bracket
  (`StepPhaseReadBegin/End` around `EnsureHostBytes`' `to_vector`,
  `src/vt/tenstorrent/tenstorrent_residency.cpp:1128`) and the launch stamp
  (`StepPhaseNoteLaunch` in `TraceReplayGraph`,
  `src/vt/tenstorrent/tenstorrent_capture.cpp:141`-area) — the completion wait
  the driver cannot see (`src/vt/tenstorrent/tenstorrent_capture.cpp`, the
  `StepPhaseState` block).
- `VT_TT_STEP_PHASES=sync`: the per-layer sampling probe — one blocking queue
  drain (`StepPhaseSyncProbe`, a persistent 1-element device tensor's
  `to_vector`) after each of the 64 layers in `DenseForwardLayers`, so each
  `[TT-STEP-PHASE] layer=...` line carries that layer's true DEVICE time.
  Host layer cost prints under the plain knob; the probe is inert while a
  capture is open (record passes report host-only). A sampling leg
  (serialized), never a wall measurement.

## The ranked cost table (steady request, 3 decode steps)

legA (`VT_TT_STEP_PHASES=1`), requests 2-4 — byte-identical to each other;
legB reproduces every row within noise (the table gives legA / legB):

| Component | legA (s/req) | legB (s/req) | share | category |
|---|---|---|---|---|
| eager cold body ×2 (host dispatch of the S=1 forward) | 67.0 | 67.0 | 64.4% | launch overhead (host, per-op dispatch) |
| — its 64-layer loop host cost ×2 | 43.4 | 43.2 | 41.7% | gdn 152-169 ms, fa 878-879 ms per layer |
| — its norm+lm_head tail host cost ×2 | 23.6 | 23.8 | 22.7% | the [1,5120]→[1,172032] keepquant head |
| trace-execution wait ×1 (the capture step's completion read) | 31.12 | 31.11 | 29.9% | replay launch overhead (device, per-command) |
| capture pass ×1 (record 2.6-4.2 s + finalize ~1.7 s) | 5.73 | 5.60 | 5.5% | capture economics |
| warms+embed+replay-launch+sampler reads | 0.1 | 0.1 | 0.1% | — |
| **sum** | **104.0** | **103.8** | **100%** | |
| measured decode wall (step lines) | 104.0 | 103.8 | | |
| measured wall via bench TPOT ×3 | 105.5-108.8 | 103.9 | | sum check ±4.5% worst case |

Genuine kernel compute (legD probe, not a wall term — it is hidden inside the
two rows above): 1.27 s per forward for the 64-layer loop (gdn 20.3 ms × 48,
fa 18.6 ms × 16), flat across requests.

The per-request step sequence both legs print:

```
cold    wall 33.2-33.9 s (body 33.1-33.9, warms 0.2-0.4 ms, embed 0.3-1.2 ms)
        gap after: 0.033-0.039 s   <- the device (1.3 s of work) is long done
capture wall 5.6-5.9 s   (cap_body 3.9-4.2 s, cap_end 1.68-1.73 s,
                          warms/embed/replay <1.5 ms)
        gap after: 31.11-31.13 s   <- the trace execution wait
cold    wall 33.1-33.2 s
        gap after: the next request's prefill (~291-300 s)
```

Prefill (outside TPOT, named for completeness): ~198-296 s/request of the
same host-dispatch wall at T=64 (4.4-4.6 s/layer host vs ~1.3 s/layer device —
the prefill kernels DO cost ~81 s/request; the host still dominates).

## The pressure axis (request 1 fresh vs request 4 near the ceiling)

**Verdict: the near-OOM retention pressure is NOT a term in the wall.** The
retention staircase does not step in this arm's decode: the per-request step
entry `dram_free` pattern repeats exactly across requests 2-4 (5628-5631 →
5748-5750 → 5059-5087 MiB; the within-request dip is transient and recovers
by the next request's entry), request 1 pays only the first capture's trace
buffers (~479 MiB at its capture step, consistent with the redesign's
436,273,152 B trace demand) plus first-pass JIT (its first cold body 53.5 s
vs 33.5 s steady), and the process holds **>5 GiB free at request 4** — the
eager arm's ~5-request OOM ceiling is not approached. Every phase row is flat
across the axis: request 4's step table equals request 2's to within noise.

The existing-instrument read-out (legF, `VT_TT_ALLOC_TRACE=1`) agrees from the
allocator side: each request's decode span produces **the identical 3,028
`[TT-ALLOC]` snapshots** with total_free min 4.93-4.98 GiB / max 5.75-5.80
GiB / last 5.74-5.79 GiB — no per-request floor drop at all. What the trace
DOES show is the per-op staging churn behind the ~32 ms/op host term: the
keepquant repair web (`kq-decode/repair`) makes **4,304 single allocations
over 100 MiB across the leg (mean 420 MiB)** — ~360 big transient planes per
decode step, allocated and freed inside the step — plus the head's chunked
`matmulbtq` staging (48 × 235 MiB). The churn is transient, repeated
identically per request, and invisible to the free-DRAM floor.

## The per-layer distribution (legD, `VT_TT_STEP_PHASES=sync`)

| pass type | host per layer | device per layer | pass totals (host/device) |
|---|---|---|---|
| decode (T=1), gdn (48) | 142-169 ms | 20.3 ms | 20.5-22.1 s / 1.27 s |
| decode (T=1), fa (16) | 857-882 ms | 18.6 ms | |
| capture record (T=1) | 41-46 ms | inert (record pass) | 2.0-2.6 s / — |
| prefill (T=64), gdn | 2.9-4.4 s | ~1.30 s | 198-296 s / ~81-82.5 s |
| prefill (T=64), fa | 3.6-5.3 s | ~1.19 s | |

The 16 full-attention layers cost ~6× a GDN layer in HOST time (857 vs 149 ms)
at equal DEVICE cost (18.6-20.3 ms) — the fa layer's op chain (RAC/PA + the
fused-preamble + qkv/o projections) is the per-op overhead's densest span, and
the norm+lm_head tail (~12.6 s host per eager step, ~38% of the body) is the
single biggest unnamed-by-layer block; both are fusion targets.

## The op census (legE, `VT_TT_TRACE_DEBUG` in-capture `[TT-OP]` entries)

**1,037 op entries per captured forward** — byte-identical across all four
captures (4 × 1037). Per forward:

| op kind | entries | op kind | entries |
|---|---|---|---|
| MatmulBTQuant | 257 | MoeSiluMul | 64 |
| MatmulBT | 240 | CausalConv1dUpdate | 48 |
| RmsNorm | 129 | GdnPostConv | 48 |
| MatmulBTQuantGrouped | 75 | GdnDecode | 48 |
| RmsNormGated | 48 | AttnQkNormRopeGate | 16 |
| CastBf16 | 16 | ReshapeAndCache | 16 |

(plus the two remaining low-count kinds; 14 distinct kinds total). 572 entries
are matmul-class — a 1-token step over 64 layers issues ~9 matmuls per layer.
At the measured walls this averages **~32 ms of host dispatch per op** on the
eager path and **~30 ms per replayed command** on the trace path — the
tt-metal dispatch cost, not the kernels (1.27 s total), is what the wall is
made of.

## Gates

- **Gate 1 (read-only instruments, tokens byte-identical with knobs unset):
  PASS.** `leg0` (knobs off) vs `legA`/`legB` (`VT_TT_STEP_PHASES=1`) vs
  `legD` (`=sync`, the serialized probe leg) vs `legE`/`legF` (the
  `VT_TT_TRACE_DEBUG` / `VT_TT_ALLOC_TRACE` read-out legs): the
  `--output-token-ids` JSON is byte-identical across all six legs — sha256
  `13c3f70b3611ff6b59fef413357704ca2bcd84251c72baff450a935de1a049cb`, stream
  `[[220,17,220,17],[220,17,220,17],[220,16,220,17],[220,16,220,16]]` (the
  recorded loop signature; its oracle adjudication stays owed to the 27B's own
  row, per the warmup redesign's issue note).
- **Gate 2 (two legs reproducible, phase deltas within noise): PASS.** legA
  vs legB, same binary: trace wait 31.107-31.126 s (0.06% spread), cold body
  33.06-33.90 s (2.5% spread, the eager host path's run variance — the bench
  TPOT band across ALL legs is 34.62-36.27 s), capture wall 5.61-5.92 s. The
  retention state recorded per leg: identical `dram_free` patterns.
- **Gate 3 (the sum check, ±10%): PASS.** Components sum to 104.0 s/request
  against the step-line wall 104.0 s (0.0% residue at the instrument level)
  and against the bench TPOT ×3 (105.5-108.8 s, -4.5% worst case — the
  client-side token delivery the step lines cannot see). No hidden residue.
- The standing preflight's `check-supported-models` FAIL
  (ClmModel/SpanExtractor/Tev1Model/XorModel docs drift) is inherited from
  the spec-commit base tree — none of those architectures is touched by this
  change; it is not this row's debt.

## Host, build, revision (the revision is part of the result)

- Host: personal Tenstorrent Blackhole P150a workstation (aarch64). Every
  device command under `flock -x $HOME/gpu.lock`; `luwen reset` + `sleep 15`
  before each leg (retry 3x); `source ~/Sources/tt/env-tt-common.sh`.
- vllm.cpp: `row/TT-27B-STEP-DECOMPOSE` (base f33dbad8c, the spec commit) +
  this instrument commit. Release Ninja build `/tmp/row-27b-decompose/build`
  (`-DVLLM_BUILD_TESTS=ON -DVLLM_CPP_TENSTORRENT=ON`, tt-metal-pin cmake
  prefix), full tree 2153/2153 rc=0; `vllm-bench` TT linkage verified (396+
  tenstorrent symbols).
- tt-metal: the recorded pin `vllm-cpp-pin/20260925` (d20b8e27f29, base
  9161e8fdb27 + the applied 4-patch series, tree byte-clean) — both the
  build-linkage tree (`~/Sources/tt/tt-metal-pin/build_release_script`) and
  the runtime tree (`~/Sources/tt/tt-metal`, verified at the same commit).
- Model: `/mnt/models/mudler-qwen3.8-27B-APEX-gguf/Qwen3.8-27B-APEX-I-Nano.gguf`
  (10.72 GB). Fixture: `tests/fixtures/tt-int8dot-sweep-sharegpt-64-20260923.json`
  (64 prompts; `--num-prompts 4` takes the first four).
- Recipe (every leg): `VT_TT_AFFINE_F32=1 VT_TT_NORM_PAD=1 VT_TT_PROGRAM_CACHE=1
  VT_DECODE_GRAPH_STATS=1` + `--num-prompts 4 --output-len 4 --concurrency 1
  --seed 0 --temperature 0 --ignore-eos --output-token-ids <leg>.json`.
  Per-leg additions: legA/legB `VT_TT_STEP_PHASES=1`; legD
  `VT_TT_STEP_PHASES=sync`; legE `VT_TT_STEP_PHASES=1 VT_TT_TRACE_DEBUG=1`;
  legF `VT_TT_STEP_PHASES=1 VT_TT_ALLOC_TRACE=1` (legF's timing is NOT used
  for the wall table — the per-op `GetMemoryView` walks perturb it; it is the
  allocator read-out only).
  c=1 is forced by the pinned tt-metal's prefill assert (BH ≤ ncores, the
  DRAM-growth row's DEVIATION 1). Logs and ids under `/tmp/27b-decompose/`.
- The undecomposed anchor re-confirmed on this tree: leg0 mean TPOT 35,157 ms
  (median 35,226) — the warmup redesign's re-derived 35,041 ms anchor's class.

## Owed (named by this table)

- The follow-up issue this row files: complete the #2469 continuation-port
  (the two missing `++s.expected_cur_pos` increments) so the captured arm
  serves replays — the 64.4% term's mechanism and the precondition for every
  captured-arm lever.
- The next perf row the table binds: the per-op dispatch overhead (~10 ms/op
  host-eager and trace-replay alike; kernels 1.3 s) — fusion/op-count on our
  side, dispatch cost on tt-metal's.
- The 27B stream's oracle adjudication (loop vs true output) — already owed to
  the 27B's own row by the warmup redesign's record; unchanged by this row.
