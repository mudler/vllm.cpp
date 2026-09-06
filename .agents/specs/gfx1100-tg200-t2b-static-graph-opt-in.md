# Spec: GFX1100-TG200-T2b — per-arch opt-in for ROCm static-graph mode

- Owning row: `GFX1100-TG200` (campaign issue #5, live landing `BACKEND-ROCM`
  #2427). This spec does not create a new row; it reworks the T2b stage of the
  existing row so the decode-graph flip is per-arch opt-in rather than
  unconditional.
- Pull request: #2777 (`row/GFX1100-TG200-T2b`).
- Base: `b9f2ef432` (`upstream/stage/ext-prs-2026-09-04`, the external-contributor
  landing branch).
- Evidence: `docs/bench-evidence/gfx1100-tg200-t2b-20260823.md`.
- Predecessor spec: `.agents/specs/rocm-decode-graph.md` (BACKEND-ROCM W1, the
  hipGraph capture seam).

## What the flip changes

`RocmPlatform::support_static_graph_mode()` flips from the base-class `false`
to `true`. This is the platform POLICY gate that, together with
`Backend::SupportsGraphCapture()` and `vt::GraphCaptureEnabled()`, admits the
`Qwen3_5DenseDecodeGraph` capture/replay path for uniform decode steps.

The flip is NOT unconditional. Mirroring the Tenstorrent pattern landed in
`src/vllm/platforms/tenstorrent.cpp` (#2910's `static_graph_requires_opt_in()`
seam), `RocmPlatform` overrides both overloads of
`static_graph_requires_opt_in()` so that only the evidence arch (gfx1100)
gets default-on capture. Every other ROCm arch keeps the pre-flip opt-in
polarity (capture off) until it carries its own committed evidence.

## Per-arch opt-in contract

The decode-graph gate in `qwen3_5_dense.cpp` is:

```cpp
const bool graph_cuda =
    platforms::GetPlatform(input.queue.device.type).support_static_graph_mode() &&
    !platforms::GetPlatform(input.queue.device.type).static_graph_requires_opt_in();
```

Three predicates conspire:

1. `support_static_graph_mode()` → `true` on ROCm. The hipGraph capture seam
   is implemented (`rocm_backend.hip`; `BeginCapture`/`EndCaptureGraph`/
   `ReplayGraph` mirror `cuda_backend.cu` call for call), and the
   mutate-src-then-replay test asserts replay never returns a snapshot.

2. `static_graph_requires_opt_in()` (no-arg, the overload
   `qwen3_5_dense.cpp` calls) → `false` on gfx1100, `true` on every other
   ROCm arch. The platform reads `vt::rocm::DeviceArchName(0)` and matches
   the `gfx1100` prefix (same prefix-match discipline
   `GcnArchNameIsGfx12PrefillWmma` uses in `rocm_arch.h`).

3. `static_graph_requires_opt_in(architectures)` (arch-aware overload) →
   delegates to the no-arg, because ROCm's scoping dimension is the GPU arch,
   not the model family. (Tenstorrent scopes by model architecture because
   its evidence families are model names; ROCm scopes by silicon because the
   evidence is a board.)

Result: on gfx1100, `graph_cuda = true && !false = true` → capture engages.
On gfx1151/gfx1103/gfx1200/gfx1201, `graph_cuda = true && !true = false` →
capture stays off, identical to the pre-flip eager path.

The framework kill switch `VLLM_CPP_CUDAGRAPH=0` (`vt::GraphCaptureEnabled()`)
still forces eager on every arch, including gfx1100 — it is read inside the
decode-graph driver's own `DenseDecodeGraphEnabled()` gate, upstream of the
platform predicates.

## Evidence on gfx1100

Recorded in `docs/bench-evidence/gfx1100-tg200-t2b-20260823.md`:

- Live capture: `[DenseDecodeGraph] captured ... padded size S=1` then
  `14 total replays across 1 captured size(s)` on a 16-token run; output
  coherent.
- A/B (256 tok × 5): 36.4 tok/s median (graph ON) vs 35.8 (same-window
  split-arm baseline) under co-tenancy contention; neutral-to-slightly-
  positive as expected. The full dispatch-gap removal (~2.08 ms/tok) shows
  only in an idle-host window (projected ~46+ from the 40.65 baseline);
  definitive idle-host capture queued as campaign follow-up.

## What stays OFF elsewhere

gfx1151 (Strix Halo), gfx1103 (Radeon 780M), gfx1200/gfx1201 (RDNA4 Navi 44):
`static_graph_requires_opt_in()` returns `true`, so the decode-graph gate
falls to the eager path. Each arch needs its own captured-arm evidence before
its opt-in flips — the same discipline Tenstorrent's `DecodeCaptureDefaultArch`
enforces for model families without committed gate pairs.

## Risks

- **Per-arch probe at call time.** `DeviceArchName(0)` is a HIP-free probe
  that reads `hipDeviceProp_t::gcnArchName` once; it is `noexcept` and returns
  an empty string when no device is present. An empty string does not match
  `gfx1100`, so a headless build degrades to opt-in (capture off) — the
  conservative answer.

- **No new env var.** The opt-in is arch-scoped, not env-scoped. The
  framework-wide `VLLM_CPP_CUDAGRAPH=0` kill switch remains the A/B and
  safety valve. Adding a per-arch env would duplicate the arch gate and
  create a second way to say the same thing.

- **Arch-aware overload delegates.** Drivers that pass `architectures`
  (e.g. `qwen3_moe_registry.cpp`) get the same answer as the no-arg overload
  because ROCm's evidence dimension is silicon, not model family. A future
  arch that needs per-model scoping can override the arch-aware overload
  independently.

## Gates

- `python3 scripts/check-env-doc.py` — green (no new env var introduced).
- `python3 scripts/check-agent-record.py` — green (no agent-record change).
- Docker HIP compile (`rocm-dev:10.0.0`, `gfx1100`) — the platform TU and
  the new self-skipping ROCm decode-graph test target compile and link.
- ROCm `ModelRegistry::Forward` decode-graph test
  (`tests/vllm/models/test_rocm_decode_graph_forward.cpp`) — self-skips
  without a ROCm device (`vt::rocm::DeviceAvailable()`); will be run on GPU
  by the operator after landing.
