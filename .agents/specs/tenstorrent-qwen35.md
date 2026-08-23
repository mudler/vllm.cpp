# BACKEND-TENSTORRENT-QWEN35: the Qwen3.5 GDN family runs on Tenstorrent — allow-list, op delta, and the first e2e gate

**Row:** `BACKEND-TENSTORRENT-QWEN35` (child of `BACKEND-TENSTORRENT`)
**Issue:** [#1715](https://github.com/mudler/vllm.cpp/issues/1715) (open; tracks the
wiring row per `tenstorrent-gdn.md` `## Owed` — "stays open until the family runs")
**Lifecycle:** `ACTIVE` (claimed 2026-08-23, helper)
**Prerequisite row:** `BACKEND-TENSTORRENT-GDN` (landed `175733000`; its eight ops are
the substrate this row makes reachable)
**Exemplar row:** `BACKEND-TENSTORRENT-MISTRAL` (`tenstorrent-mistral.md`) — the third
arch wired onto TT and the e2e recipe this row mirrors

## Now

`ACTIVE`. Spec committed spec-first 2026-08-23 (this commit) with the matrix row and
the two checker re-pins. Owed: W0 refusal sweep, W1 op delta, W2 e2e gate, W3 the
GDN-row reviewer leftovers. Nothing implemented yet.

## Scope

Make `Qwen3_5ForConditionalGeneration` (the dense text-only Qwen3.5 GDN hybrid)
run end to end on the Blackhole P150, and prove it:

1. **Allow-list.** Add the arch to `TenstorrentPlatform::supports_model_architecture`
   (`src/vllm/platforms/tenstorrent.cpp`). The P150 is discrete — an op miss refuses
   by name, never falls back to CPU — so the allow-list entry lands LAST in the
   implementation order, after the op delta is complete.
2. **Op delta (W1).** Complete exactly the op set the 0.8B forward refuses by name on
   TT (W0 pins the list empirically). Named candidates from source inspection:
   - `kGdnPostConv` — direct dispatch, no composite fallback
     (`src/vt/ops.cpp:4270-4292`; model sites `qwen3_5.cpp:4146`, `:4600`, `:4980`).
   - `kSigmoidGateBf16` — direct dispatch, no composite fallback
     (`src/vt/ops.cpp:4142-4154`; model site `qwen3_5.cpp:2546`).
   - `kAttnQkNormRopeGate` — the GDN full-attention preamble. The model checks
     `OpRegistered` first (`qwen3_5.cpp:5192`, `:5209`, `:5317`); unregistered it
     takes the unfused path (`AttnGateSplit` + `RmsNorm` + rope), and `AttnGateSplit`
     is ALSO unregistered on TT — so W1 registers the fused op (the form the model
     prefers and the ROCm gate's BACKEND PROOF names), not the unfused pair.
   - `kAttnQkNormRope` — the plain (non-gated) preamble, same
     registration-guarded shape; same decision.
   The sweep is authoritative: if it refuses an op this list does not name, that op
   is in scope too; if a candidate is not refused, it is not implemented.
3. **First e2e gate (W2).** The TT device-golden treatment for
   `tests/parity/test_qwen35_paged_engine.cpp`, mirroring
   `test_mistral_paged_engine.cpp`: `VT_DUMP_IDS=1` bootstrap on the P150 →
   `our_ids_tenstorrent.i32`; transformers teacher-forced near-tie gaps via
   `scripts/qwen3-neartie-gap-transformers.py` (the ratified POL-ORACLE deviation,
   same tool and precedent as the Qwen3-0.6B and Mistral-7B TT goldens); commit the
   TT golden pair; full gate 16/16 PASS with near-tie ≤500 milli-nats, strict
   token-exact reported, and BACKEND PROOF (GDN op set selections > 0, declines == 0).
4. **Capacity decision (recorded, not deferred).** `Qwen/Qwen3.5-0.8B` bf16
   ≈1.6 GB fits the P150 trivially (proven envelope: Mistral-7B bf16 ≈14.5 GB).
   Qwen3.8-27B bf16 ≈53.8 GB does NOT fit, and the GGUF k-quant arms have no TT
   kernels, so the 27B-on-TT lane is refused by name and owed (see `## Owed`).
   The first family gate is therefore the 0.8B bf16 arm — the cheapest checkpoint
   that exercises the complete GDN chain end to end.
5. **Reviewer leftovers from the GDN row (W3).** (a) The d2h traffic counter misses
   the `EnsureGdnCacheDevice` slow-path download and the `CommitConvTransposed`
   untracked-buffer fallback; (b) `EnsureGdnCacheDevice`'s fast path does not check
   `conv_transposed`, so a host pointer reused across roles would confuse
   geometries (`qwen3_5.cpp` uses distinct buffers today, so this is a hardening
   with a test, not a live bug).

**Excludes:** the MoE arches (`Qwen3_5MoeForCausalLM`,
`Qwen3_5MoeForConditionalGeneration` — no TT MoE kernels, no fitting checkpoint);
GGUF k-quant arms (no TT kernels; refused by name — owed); the VL tower; 27B e2e;
`src/vt/cpu/` (the oracle stays untouched) and `src/vt/cuda/`; `#1625` capture and
`#1627` async readback (independent, as in the GDN row).

## Upstream chain

No vLLM Tenstorrent implementation exists (`vllm-omni` does not register TT either);
`tt-forge` is the registered secondary oracle for the hardware but the model-level
behavior reference is vLLM as everywhere. Two oracle lanes, both ratified:

- **Op level:** our own CPU f32 arm (`src/vt/cpu/cpu_ops.cpp`) — the same
  residual-golden precedent the GDN row used.
- **E2e level:** the pinned vLLM-ROCm greedy capture is the base golden
  (`tests/parity/goldens/qwen35_greedy_0_8b/manifest.json`, vLLM `555967922`,
  model revision `2fc06364715b967f1860aea9cf38778875588b17`), but the gate compares
  TT prefixes, so the near-tie gaps are re-derived teacher-forced via
  `scripts/qwen3-neartie-gap-transformers.py` — the exact deviation the Mistral row
  recorded and the Qwen3-0.6B TT golden ratified before it.

## Our baseline

TT registers 27 ops today (`tenstorrent_ops.cpp:4434+`), including the GDN row's
eight. The Mistral row proved the full e2e recipe on this exact board: bootstrap →
transformers gap (max 0.0625 nats there) → committed golden pair → 16/16 with
BACKEND PROOF, tolerating the known exit-139 MeshDevice teardown (#1486) after the
green summary.

## Port map

| Op | Contract | CPU oracle | Model sites | TT realization |
|---|---|---|---|---|
| `kGdnPostConv` | `ops.cpp:4270-4292` (g/beta/a_log/dt_bias f32; a/b f32-or-bf16 views) | `cpu_ops.cpp:3800` | `qwen3_5.cpp:4146`, `:4600`, `:4980` | ttnn eltwise composition (sigmoid/exp/mul chains), the GDN row's elementwise pattern |
| `kSigmoidGateBf16` | `ops.cpp:4142-4154` (attn f32/bf16, gate f32, out bf16) | `cpu_ops.cpp` (grep `kSigmoidGateBf16`) | `qwen3_5.cpp:2546` | ttnn sigmoid + mul + cast |
| `kAttnQkNormRopeGate` | `ops.cpp` recipe `:1124-1136`; wrapper `AttnQkNormRopeGate` | CPU standalone op | `qwen3_5.cpp:5220`, `:5378` | fused: gemma-RMSNorm(q,k) + partial NeoX rope + gate split, one launch or the row's fused pattern |
| `kAttnQkNormRope` | FusedChain fast realization `ops.cpp` (`DispatchFusedFast`) | CPU standalone op | full-attn layers (guarded) | same shape minus the gate |

The implementer pins exact CPU line anchors when writing each kernel; the contract
column above is the entry point, not the whole chain.

## Design

- **Fused over unfused for both preambles.** The model already prefers the fused op
  when registered; the unfused path needs `AttnGateSplit`, which would be a fifth
  kernel for no benefit. One fused kernel each mirrors the registration-guard
  semantics and keeps the BACKEND PROOF list identical to the ROCm gate's.
- **bf16 storage, f32 intermediates** exactly as the GDN row: model-path buffers
  stay the model dtype; an f32 intermediate names its reason inline.
- **Allow-list lands last.** The discrete device's by-name refusal is the safety
  property; the arch string only reaches the allow-list when the sweep is empty.
- **Host-free decode stays default-on** (`VT_TT_HOST_FREE_DECODE` unset); the e2e
  gate runs ambient plus the `=0` leg, as the GDN row did, so the flip is exercised
  both ways at e2e depth, not just op depth.
- **Weights.** `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`
  (verified 2026-08-12 via download metadata, `qwen35_greedy_0_8b/manifest.json`),
  bf16, ~1.6 GB. Download to the local HF cache authorized 2026-08-23.
  `docs/USAGE.md` gains the TT arm entry (repo, revision, size) in the landing
  change, with the refused arms named beside it.

## Tests to port

- Op-level doctest cases in `tests/vt/test_tenstorrent_backend.cpp`, one per new op,
  `CompareVsOracle` against the CPU f32 arm, in the GDN row's case style (shape
  sweep + dtype arms + NaN-hardened comparator).
- Registration-sweep case: every op the 0.8B forward dispatches has a TT
  registration (the empty-refusal property, asserted by name list).
- The TT arm of `test_qwen35_paged_engine.cpp`: device-golden pair load, near-tie
  methodology, BACKEND PROOF assertions — mirroring `test_mistral_paged_engine.cpp`.
- W3: counter/hardening tests per leftover.

## Gates

1. **Focused:** new op cases green on the P150, both ambient legs
   (`VT_TT_HOST_FREE_DECODE` unset and `=0`).
2. **E2e:** 16/16 prompts PASS on the P150, near-tie ≤500 milli-nats, strict-exact
   reported, BACKEND PROOF selections > 0 and declines == 0 for the GDN op set, both
   ambient legs. Exit 139 after the green summary is the known #1486 teardown and
   counts green.
3. **Full TT suite green; CPU gate green; `scripts/agent-preflight.sh` all-green.**
4. **Mutation evidence** per asserted guarantee, re-run by the fresh reviewer.

## Dependencies

- The pinned tt-metal checkout and the GDN row's adapter/shadow substrate
  (unchanged by this row).
- Real Blackhole P150 (`thalia`) under the file mutex; not an `rc` fleet device.
- The 0.8B checkpoint in the local HF cache (authorized).
- A python env with torch+transformers for `qwen3-neartie-gap-transformers.py`
  (the Mistral row used torch 2.7.1+cpu / transformers 5.8.1; reuse or rebuild).
- Independent of #1625/#1627.

## Work breakdown

- **W0 — sweep (pins W1).** Scratch-wire the allow-list, run the 0.8B forward on
  the P150, collect every by-name refusal. Recorded in `## Evidence`. Not committed
  as product; the sweep's op list is the W1 scope contract.
- **W1 — op delta.** Implement each refused op as a TT kernel + oracle case +
  mutations, the GDN row's method.
- **W2 — e2e.** Allow-list entry (one line, final commit of the wave), bootstrap,
  transformers gap, committed TT golden pair, full gate, `docs/USAGE.md`.
- **W3 — leftovers.** d2h counter completeness; `conv_transposed` fast-path check;
  tests for both.

Each wave lands focused-green before the next; the full gate + fresh review close
the row.

## Risks

- **Cold JIT window.** 0.8B is smaller than Mistral-7B's cold run but larger than
  Qwen3-0.6B's ~30 min; the e2e bootstrap needs an uninterrupted board window with
  a warm cache on retries.
- **Near-tie deviation.** The gap golden is transformers-teacher-forced, not
  vLLM-ROCm — ratified twice already; the spec records it a third time rather than
  silently inheriting.
- **A real divergence** (outside the near-tie band) is an open gap: report, do not
  tune the band. Stop condition.
- **Host-free decode interplay** with the new elementwise ops (capture stays
  opt-out; ambient and `=0` legs both gate).

## Stop conditions

- Any need to edit `src/vt/cpu/cpu_ops.cpp` (the oracle) — out of scope, escalate.
- A non-near-tie e2e divergence that does not reproduce on the CPU arm.
- The P150 or the checkpoint download becoming unavailable mid-row (gate stays
  PENDING with the named blocker; never waived silently).

## Owed

- **GGUF k-quant arms for the family on TT** — no TT kernels; refused by name.
  A standing family requirement (AGENTS `## Shared seams`), not a choice.
- **Qwen3.8-27B on TT** — no arm fits the P150 (bf16 53.8 GB; quant kernels owed
  above). Refused by name at load.
- **MoE arches / 2.4T lane** — TT MoE kernels do not exist.
- **GDN under capture** — unchanged from the GDN row's `## Owed` (#1625 first).
- On landing, the GDN row's lifecycle moves (`ACTIVE` → `DONE` + `## Outcome`) in
  the same change: its ops become production-reached.

## Git integration

One pull request for spec and implementation (row claim answer 2026-08-23, recorded
in `.agents/developer-preferences.md`). Base `origin/main` @ `175733000`. Branch
`row/BACKEND-TENSTORRENT-QWEN35`, worktree `/home/lu_zero/Sources/vllmcpp-tt-qwen35`.
