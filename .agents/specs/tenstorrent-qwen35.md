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

`ACTIVE`. Spec committed spec-first 2026-08-23 with the matrix row and the two
checker re-pins. **W0 (refusal sweep) and W1 (op delta) landed** — see `##
Evidence`; the sweep's scratch allow-list and CPU stubs are reverted out of
the tree, so the arch is still refused at load until the W2 allow-list entry.
Owed next: W2 e2e gate, W3 the GDN-row reviewer leftovers.

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
- **`kCausalConv1dUpdate` production bf16 arm** — capability refusal (W0 item 4):
  TT already computes the bf16 cache via f32 shadows; enabling it needs the
  `SupportsCompressedConvState()` flip in `src/vt/tenstorrent/tenstorrent_backend.cpp`,
  outside the W1 delegated file set. Escalated to the operator 2026-08-23; blocks the
  W2 e2e gate on the default (bf16-state) arm, not on `VT_GDN_STATE_BF16=0`.
- On landing, the GDN row's lifecycle moves (`ACTIVE` → `DONE` + `## Outcome`) in
  the same change: its ops become production-reached.

## Git integration

One pull request for spec and implementation (row claim answer 2026-08-23, recorded
in `.agents/developer-preferences.md`). Base `origin/main` @ `175733000`. Branch
`row/BACKEND-TENSTORRENT-QWEN35`, worktree `/home/lu_zero/Sources/vllmcpp-tt-qwen35`.

## Evidence

All board runs on the P150 (`thalia`) inside the `${GPU_LOCK:-$HOME/gpu.lock}` file
mutex, `TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal` (pinned tree), build
`ninja -C build tests/test_tenstorrent_backend`. Exit 139 after a green doctest
summary is the known #1486 teardown, not a gate failure.

### W0 — refusal sweep (runs 1-8, `/tmp/w0_sweep_run{1..8}.log`)

Scratch-wired the arch allow-list (reverted before commit; `src/vllm/platforms/
tenstorrent.cpp` carries no Qwen3.5 entry in this change). Refusals, in sweep
order:

1. `kGdnPostConv` — `no kernel for op GdnPostConv` (run 1).
2. `kAttnQkNormRopeGate` — unregistered on TT, reached 12x.
3. `kSigmoidGateBf16` — unregistered on TT, reached 12x.
4. `kCausalConv1dUpdate` — capability refusal on the PRODUCTION bf16 arm
   (`conv_state` bf16 + TT `SupportsCompressedConvState()`=false, ops.cpp:1721).
   The TT kernel already handles bf16 caches via f32 shadows; the fix is the
   backend-flag flip in `tenstorrent_backend.cpp`, OUTSIDE this task's file set —
   **escalated to the operator** (owed below).

Kernel DEFECT surfaced (not a refusal): `ScatterRowsExact` — TILE rank-4 reshape
inflated physical volume (32 GiB OOM, runs 2-4); ROW_MAJOR reshape overflowed L1
circular buffers (runs 5+7). Fixed in W1 (below).

NOT refused, because never dispatched (fused inside the two compositions above):
`kAttnQkNormRope`, `kGdnGBeta`, `kGdnConvSplit`, `kAttnGateSplit`. Their W0 CPU
sweep-stubs are deleted; no TT kernels owed.

Sweep completion: run 8 (f32 state arm, CPU stubs) generated 2 tokens end to end
(`first=220`). Sweep reach counts: GdnPostConv 36, GdnStateScatter 36, GdnDecode
18, AttnQkNormRopeGate 12, SigmoidGateBf16 12.

### W1 — op delta (all focused logs under `/tmp/w1_*.log`)

Three new TT kernels in `src/vt/tenstorrent/tenstorrent_ops.cpp`, each red-first
(Registration removed → `no kernel for op ...` refusal captured), then green
against the CPU f32 oracle, then negatively mutated and restored:

- `SigmoidGateBf16Kernel` — red `/tmp/w1_red_sigmoidgate.log` (exit 1, `no kernel
  for op SigmoidGateBf16 (id 64)`), green `/tmp/w1_green_sigmoidgate.log`:
  16/16 configs, `max_abs=0` (SFPU sigmoid rounds to the same bf16 values as
  the CPU oracle in every tested element).
- `GdnPostConvKernel` — red `/tmp/w1_red_gdnpostconv.log` (id 71), green
  `/tmp/w1_green_gdnpostconv.log`: 100/100. First green attempt used one-shot
  `ttnn::softplus` and FAILED 5/100 at `g max_rel 4.4e-4` (SFPU poly ~1e-6
  ABSOLUTE fit error); the committed form composes `relu(x) +
  log1p(exp(-|x|))` in f32 (`≤1.9e-7` rel, reproduces the threshold-20 branch).
- `AttnQkNormRopeGateKernel` — red `/tmp/w1_red_attnqknormropegate.log` (id 73,
  op_provider.cpp:563), green `/tmp/w1_green_attnqknormropegate.log`: 6/6 configs
  (T=1/3/65, Hq/Hkv GQA 4:2 and 32:8, rot 64/128, gemma on/off, f32/bf16 in),
  q/k `max_abs ≤ 5.5e-4` (envelope 0.02), gate `max_abs=0` (exact passthrough).
  A first implementation sliced+reshaped ONE `[T, Hq*2Dh]` upload on device; the
  q/gate legs returned full-scale wrong data while the full-width k leg was
  correct — slice+reshape on a fresh `from_vector` TILE upload is unsafe at this
  pin. Committed form: the q|gate split happens in the host gather; every leg
  uploads already-shaped (no device slice/reshape).
- `ScatterRowsExact` zero-copy fix (ROW_MAJOR convert + `ttnn::experimental::view`
  `[slots,1,1,cols]` + indexed_fill dim=0, ~256 KB): verified on the REAL kernel
  paths (`/tmp/w1_evidence_scatter_decode_roundtrip.log`): GdnStateGather/Scatter
  `max_abs=0` (B1), GdnDecode-vs-oracle (B2) and the Prefill<->Decode final-state
  round-trip (B3) both exit 0. `kGdnDecode`/`kGdnStateScatter` now register the
  REAL kernels; the W0 CPU sweep-stubs are deleted from the tree.

Negative mutations (each focused gate red, then restored byte-for-byte;
`/tmp/w1_mutations_m1_m2.log`, `/tmp/w1_mutations_m3_m4.log`):

- M1 sigmoid dropped from SigmoidGateBf16 → 16/16 assertions failed
  (`max_rel 1.5e4`), exit 1.
- M2 composed softplus → one-shot `ttnn::softplus` (the historical defect) →
  95/100, 5 failed at `g max_rel 4.4e-4`, Status FAILURE.
- M3 NeoX rope sign flip (`subtract`→`add`) → q/k failed in all 6 configs.
- M4 gate round-tripped through bf16 → gate failed the exact-passthrough check
  on every f32-in config (`max_abs 0.00195` = one bf16 ULP).

Full gate (no scratch stubs, allow-list reverted) — BOTH legs GREEN:
`/tmp/w1_fullgate_leg1.log` (ambient) and `/tmp/w1_fullgate_leg2.log`
(`VT_TT_HOST_FREE_DECODE=0`): 36/36 test cases, 2259/2259 assertions,
`Status: SUCCESS!` on each (exit 139 = #1486 teardown after the summary).
