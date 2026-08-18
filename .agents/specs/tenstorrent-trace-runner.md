# Tenstorrent mesh-trace decode capture — feasibility spike

Status: **DRAFT, 2026-08-12.** A read-only feasibility spike (POL-SPIKE-FIRST,
POL-NO-CEILING). No code change is in scope until the decision (§Risks/decisions)
is recorded and the maintainer accepts the tradeoff.

Proposed row id: `BACKEND-TENSTORRENT-TRACE-RUNNER` (child of
`BACKEND-TENSTORRENT`; the graph-capture *foundation* already landed as #354 /
`59568772` — this row is the *runner wiring* the handoff §8/§9 names as "not
done").

## Scope

**In.** Establish the facts needed to decide whether wiring the landed
graph-capture surface into a Qwen3-dense / Mistral **decode** graph on
`kTENSTORRENT` is feasible and worth it, and at what cost. Three questions:

1. **Is there a host-free decode region to capture?** At pure decode (T=1),
   do the current hybrid thresholds route any op to host, breaking ttnn
   `begin_trace_capture` (which requires a contiguous device-op region with
   no host sync / `to_vector` readback in the middle)?
2. **What does capture cost in tok/s?** If forcing the host-routed ops
   (RoPE, residual RMS) all-device at T=1 is the prerequisite for capture,
   does the resulting capture *recover* the tok/s that all-device-at-T=1
   loses? The handoff §6 records that "always device residual/RoPE"
   regressed Qwen3-0.6B ~12.3→10.5 tok/s; capture must beat 10.5 to be
   worth it, and ideally approach/beat 12.3.
3. **Does ttnn program-cache warm-up work before `begin_trace_capture`?**
   ttnn requires the same op shapes to be JIT-compiled (program-cache warm)
   before capture begins (ttnn `graph_query_op_runtime.hpp` pattern). Is
   that satisfied by one eager forward, as DeepSeek-V2's CUDA path does
   (`deepseek_v2.cpp:1028-1034`)?

**Out.** No implementation, no perf claim, no model expansion. If the
decision is "go," the implementation is a follow-on under this row, not a
new row. Prefill capture and multi-token-chunk capture (where thresholds
already go all-device) are explicitly separate — this spike is about
**decode (T=1)**, the handoff's named target.

## How the other backends do it (POL-MIRROR-VLLM / POL-SEAM-RUNNER)

Surveyed on `origin/main` (`a89b3c45`):

- **CUDA (`src/vt/cuda/cuda_backend.cu:198-240`)** — the ONLY backend with
  `SupportsGraphCapture() == true`. Capture contract (cuda_backend.cu:184-197):
  every op in the region runs ASYNC on the stream (no `Synchronize`, no
  host↔device blocking copy); NO `cudaMalloc`/`cudaFree` inside the region
  (pool pre-warmed, every alloc a pool hit); captured pointers stay fixed
  across replays, only contents change (written by an async copy BEFORE
  Replay). `BeginCapture`→`cudaStreamBeginCapture`,
  `EndCaptureGraph`→`cudaStreamEndCapture`+`cudaGraphInstantiate`,
  `ReplayGraph`→`cudaGraphLaunch`.
- **Metal (`metal_backend.mm:13-15`), Vulkan (`vulkan_backend.cpp:16-18`)** —
  both `SupportsGraphCapture() == false`, with comments naming the eventual
  mapping (`MTLIndirectCommandBuffer` / pre-recorded `VkCommandBuffer`) and
  explicitly NOT implementing it. So TT would be the first non-CUDA backend
  to ship capture.
- **Model-side capture site** (`deepseek_v2.cpp:1028-1034`,
  `qwen3_moe.cpp:506-509`, `qwen3_dflash.cpp:1091-1095`): the region
  captured is `ForwardLayers` — the FULL layer stack, device-resident. The
  pattern is: embed input → `BeginCapture` → `ForwardLayers` →
  `EndCaptureGraph` → `ReplayGraph` per step. One eager step first warms the
  pool/residency/kernel cache (the "cold size" path at `deepseek_v2.cpp:1051`).

**Implication for TT:** the CUDA contract maps almost 1:1 onto ttnn trace
capture's own requirements (contiguous device region, no host readback,
pre-warmed program cache). The TT backend's capture surface
(`tenstorrent_backend.cpp:70-76`) already maps `BeginCapture`→
`TraceBeginCapture` etc. onto `ttnn::begin_trace_capture`/`end_trace_capture`/
`replay_trace`. What is NOT done is the model-side wiring: no dense forward
(Qwen3/Mistral) calls `BeginCapture`/`ForwardLayers`/`EndCaptureGraph` on TT
today, and the decode path currently interleaves host ops.

## Upstream chain

**No upstream vLLM equivalent.** vLLM's CUDA-graph capture is the loyal
contract (`vllm/v1/worker/gpu/worker.py::capture_model`, already mirrored by
this tree's CUDA path). ttnn's trace API is the dependency-chain leg:
`ttnn::begin_trace_capture(device, cq_id)` / `end_trace_capture` /
`replay_trace` (tt-metal `ttnn/cpp/ttnn/trace.hpp`), already wired in
`tenstorrent_ops.cpp::TraceBeginCapture/TraceEndCapture/TraceReplay`.

## Our baseline (the blocker, precisely)

At pure decode (T=1) for Qwen3-0.6B (Hq=16, Hkv=8, D=1024), the current
hybrid thresholds in `tenstorrent_ops.cpp`:

- **Residual RMS** (`RmsNormKernel`, line 1070): `kDeviceResidualMinRows =
  32`; at T=1, `rows=1 < 32` → the **host** f32 path runs (bit-identical to
  CPU; verified by `BACKEND-TENSTORRENT-RESIDUAL-GOLDEN`).
- **RoPE** (`PreferDeviceRope`, line 1342): `tokens * heads >= 64`; at T=1,
  `1*16 = 16 < 64` → the **host** RoPE path runs.

So **every decoder layer at T=1 hits the host twice** (residual + RoPE),
interleaved with device matmuls/attention. There is no contiguous
device-only region spanning a full layer, let alone the layer stack. ttnn
capture aborts on the host readback in the middle — this is exactly the
"needs a region free of host RoPE/residual" note in the handoff §8.

The residual-golden row measured the *numerics* of flipping residual to
device at rows≥32; the *perf* of flipping both residual AND RoPE to
all-device at T=1 is what this spike must measure. The handoff §6 says that
flip ("always device residual/RoPE") regressed Qwen3-0.6B ~12.3→10.5 tok/s.
Capture's value proposition is collapsing the per-step host-API overhead;
whether it recovers the ~1.8 tok/s loss AND goes beyond 10.5 is the open
question.

### Spike finding: the decode runner is a shared framework, gated on TWO methods

The dense decode-graph framework already exists and is **model-shared**:
`Qwen3DenseDecodeGraph` + `DenseDecodeGraphForward` in
`src/vllm/model_executor/models/qwen3.cpp:489,670`, used by Qwen3, Mistral,
Llama, and InternLM2 (all four registries call `DenseDecodeGraphForward`).
Its `Impl` ctor (`qwen3.cpp:495-497`) gates enablement on BOTH:

```cpp
enabled = env_on &&
          platforms::GetPlatform(...).support_static_graph_mode() &&
          b.SupportsGraphCapture();
```

- `SupportsGraphCapture()` — TT returns `true` (landed #354).
- `support_static_graph_mode()` — the `Platform` method
  (`include/vllm/platforms/interface.h:189`, base default `false`). **TT
  does NOT override it** → inherits `false` → the decode-graph framework is
  **disabled** for TT today, even though the backend can capture. Only CUDA
  overrides it to `true` (`cuda.cpp:59`); ROCm explicitly stays `false`
  (`rocm.cpp:67`, "hipGraph is the mapping and is not wired").

**Implication:** wiring TT into the decode runner is, at the platform
seam, a one-line override (`support_static_graph_mode() == true`). But that
alone is insufficient: the framework would then attempt to capture the T=1
decode forward, which (per the thresholds above) hits host RoPE+residual
every layer and would abort ttnn capture. So the real prerequisite is a
host-free decode region; the platform flag is the *enabler*, not the work.

This is the same shape as CUDA's path: CUDA's decode-graph capture works
because the CUDA ops are all-async (no host sync in the region). TT's
decode currently isn't all-device at T=1, so capture can't apply yet.

## Work breakdown (spike-only, read-only except for measurement)

1. **Confirm the host interleaving** (static): trace the T=1 decode op
   sequence through `Qwen3DenseModel::Forward` and list every `EnsureHost` /
   host-path hit per layer. Output: a per-layer host-touch map.
2. **Measure all-device-at-T=1 baseline** (on-card, temporary env override):
   add a hidden `VT_TT_FORCE_DEVICE` escape hatch (local only, NOT shipped)
   that forces `kDeviceResidualMinRows=1` + `PreferDeviceRope=true` always,
   run the Qwen3-0.6B `vllm-cli` smoke, record warm tok/s. Compare to 12.3
   (hybrid) and 10.5 (handoff's "always device" number).
3. **Probe ttnn program-cache warm-up**: confirm one eager forward makes the
   capture-region op shapes resident in ttnn's program cache (the
   `begin_trace_capture` precondition). Static + a tiny standalone probe if
   needed.
4. **Decision record** (§Risks/decisions): go / no-go / go-only-for-prefill.
   No implementation in this spike.

## Gates

**No correctness gate owed by the spike itself** — it produces a decision
record, not a shipped change. The temporary `VT_TT_FORCE_DEVICE` measurement
is a local throwaway, gated behind an env var that never ships; it does not
alter the committed decode path. Any implementation follow-on carries its
own token-exact / distributional gate against the oracle (Qwen3-0.6B
`our_ids_tenstorrent.npy` / the Mistral TT golden pair).

**Hardware:** real Blackhole (P150) for the measurement step.

## Dependencies

- `BACKEND-TENSTORRENT` (parent) — `ACTIVE`.
- The landed trace foundation (`59568772`, #354): `SupportsGraphCapture`,
  `BeginCapture`/`EndCapture`/`Replay`, `EndCaptureGraph`/`ReplayGraph`/
  `DestroyGraph`, unit-tested (matmul warm→capture→replay×3, max_abs=0).
- Qwen3-0.6B checkpoint (already on this box) for the smoke measurement.

## Risks/decisions

- **The likely outcome is "no-go for pure decode (T=1), go for prefill /
  multi-token."** The hybrid thresholds exist *because* all-device-at-T=1
  lost ~1.8 tok/s; if capture doesn't recover that plus margin, pure-decode
  capture is a net loss. But at prefill / chunked-prefill (T≥32), the
  thresholds ALREADY go all-device — that region IS host-free today and is
  the natural first capture target. The spike should not assume decode is
  the right capture scope; prefill may be where TT capture first pays off.
- **ttnn capture preconditions are stricter than CUDA's in one way:** ttnn
  needs the *exact op shapes* program-cache-warm before capture (not just
  pool-pre-warmed). One eager forward may suffice for a fixed-shape decode
  graph, but a batched/padded decode (variable B) would need a capture per
  padded size — same as CUDA's multi-graph path (`EndCaptureGraph` handle
  API), which the TT backend already exposes.
- **Not a correctness risk:** capture/replay is numerically identical to
  eager (the landed unit test proves max_abs=0 on replay). The risk is
  purely perf (does it beat the hybrid baseline?) and scope (decode vs
  prefill).

## Outcome (2026-08-12/13) — Q1+Q2 answered, capture attempted, **NO-GO measured**

**Q1 (host-free region?):** NO at pure decode (T=1). Static trace: `RmsNormKernel`
(rows=1 < 32 → host) and `PreferDeviceRope` (T*H = 16 < 64 → host) both route
to host every layer. Plus the shared decode-graph framework
(`Qwen3DenseDecodeGraph`) is disabled for TT because the platform does not
override `support_static_graph_mode()` (base default `false`; only CUDA
returns `true`).

**Q2 (all-device-at-T=1 tok/s cost?):** measured on real Blackhole P150,
Qwen3-0.6B, `vllm-cli --prompt Hello --max-tokens 4 --repeat 3`, with a
local-only `VT_TT_FORCE_DEVICE` override forcing both thresholds all-device:

| config | warm tok/s |
|--------|-----------|
| hybrid (current default) | **12.5** (12.49, 12.54) |
| all-device (VT_TT_FORCE_DEVICE=1) | **10.7** (10.77, 10.66) |

Reproduces the handoff §6 number precisely (~12.3→10.5; 12.5→10.7 here —
same ~1.8 tok/s / ~14% regression). The throwaway override was reverted.

### Q2b — capture attempted on-card: ABORTS (the decisive measurement)

The earlier "no-go" was a guess (capture "implausibly" recovers 1.8 tok/s).
To actually decide, the spike ran the capture experiment: local-only flips of
both `support_static_graph_mode()` → `true` (enables `Qwen3DenseDecodeGraph`,
which wires `BeginCapture`/`ForwardLayers`/`EndCaptureGraph` on the captured
padded-batch slot) AND `VT_TT_FORCE_DEVICE` (RoPE+residual all-device), then
the same Qwen3-0.6B cli smoke.

**Result: capture ABORTS.** ttnn raises `TT_FATAL: Reads are not supported
during trace capture` with a backtrace through `ttnn::Tensor::to_vector<float>`
— a device→host readback fires inside the captured `ForwardLayers` region.
`[Qwen3DenseDecodeGraph] dense decode graph: 0 total replays across 1 captured
size(s)` confirms the graph never successfully replayed.

**This is strictly stronger than the tok/s guess.** Even with the two hybrid
thresholds forced all-device, the T=1 forward still performs `to_vector` host
readbacks (the `DownloadToHost`/`EnsureHost` path in ops — embedding result,
paged-attention output, or logits), and `ttnn::begin_trace_capture` prohibits
*any* host read during the captured region. So capture does not merely
*start from a 1.8 tok/s deficit* — it **cannot run at all** on the current
T=1 forward without first eliminating every `to_vector` readback in the
captured region, which is a much larger redesign than flipping two thresholds.

Both local overrides (`support_static_graph_mode`, `VT_TT_FORCE_DEVICE`) were
reverted; no code shipped.

**Q3 (ttnn program-cache warm-up?):** moot — capture aborts before warm-up
matters.

### Decision

**NO-GO for pure T=1 decode capture — now measured, not assumed.** The T=1
forward does `to_vector` host readbacks that ttnn trace prohibits; forcing
the two hybrid thresholds all-device is insufficient because other ops still
read back. Decode capture requires a host-free `ForwardLayers` (every op
device-resident end-to-end, no `to_vector`), which is a redesign of the TT
forward's host-staging model, not a threshold tweak.

**Open follow-on (separate row): prefill / multi-token chunk capture.** At
T≥32 the thresholds already go all-device; whether the prefill region is also
free of `to_vector` readbacks is the open question Q3 should answer next.
The value proposition there is different (prefill has more host-API overhead
per step) and the host-readback constraint is the same ttnn rule, so the
prefill row must first audit its readbacks before claiming capture is
feasible.

## Now

`SPIKE`. NO-GO for pure T=1 decode capture: the T=1 forward does `to_vector`
readbacks that ttnn trace prohibits. Decode capture moved to
`BACKEND-TENSTORRENT-HOST-FREE-FORWARD` (#1105). Next for this row: audit
`to_vector` in the prefill / multi-token region before claiming capture
there.
