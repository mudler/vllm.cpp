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
checker re-pins. **W0 (refusal sweep), W1 (op delta), and W2a are done** — see
`## Evidence`. W2a flipped `SupportsCompressedConvState()` /
`SupportsCompressedGdnState()` (the production bf16 mamba-cache arms), pinned
both arms against the CUDA bf16-STORAGE emulation, added the arch allow-list
entry, and fixed the `ScatterRowsExact` L1 overflow that killed the first e2e
bootstrap (`SplitFactor` born-split shadows). Owed next: W2b — teacher-force
the dumped TT ids via `scripts/qwen3-neartie-gap-transformers.py`, commit the
TT golden pair, run the full 16/16 gate with BACKEND PROOF, and `docs/USAGE.md`;
then W3, the GDN-row reviewer leftovers.

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

### W2a — bf16 cache arms + allow-list + the L1 scatter fix (all focused logs
### under `/tmp/w2a_*.log`)

**The capability flips** (`src/vt/tenstorrent/tenstorrent_backend.cpp`):
`SupportsCompressedConvState()` and `SupportsCompressedGdnState()` now return
true. The TT kernels compute through f32 shadows and honor bf16 STORAGE at the
boundary: every committed shadow value re-rounds through bf16 on device (RNE
typecast round-trip, zero PCIe), mirroring CUDA's "read/written in f32
registers" (`cuda_backend.cu:119`, `cuda_gdn.cu`). This states as a capability
what `CheckConvCommon` asks; it un-refuses the W0 item-4 production arm.

Red-first / mutation evidence for the flips and their oracle arms:

- RED `/tmp/w2a_red_conv_bf16.log`: with the flip reverted, the bf16 conv_state
  arm refuses by name (the ops.cpp compressed-state check). GREEN
  `/tmp/w2a_green1_conv_bf16.log`, `/tmp/w2a_green2_conv_bf16.log`: the
  bf16-state arm matches the CUDA bf16-STORAGE emulation — step 1 exact,
  steps 2+ need the per-step store rounding (without it a full-mantissa x tap
  diverges in step 2+; with it `max_abs ≈ 5.8e-4` full-mantissa arm and
  bit-exact values where x is bf16-representable, the production activation
  dtype).
- RED `/tmp/w2a_red_gdn_bf16.log`: gdn_decode refused for a bf16 state before
  the flip ("state must be f32, or fp16/bf16 on a backend whose GDN kernels
  support a compressed state", ops.cpp:1783). GREEN `/tmp/w2a_green_gdn_bf16.log`:
  per-step outs match the f32 path with host-side bf16 round-trips bit-for-bit
  after the commit-time rounding.
- Reviewer LOW-a/b repairs: std::string labels, out_bf16 arms, and the
  kAttnQkNormRopeGate envelope tightened to `1e-3` f32-out with M5 redone —
  `/tmp/w2a_green_lowb.log`, `/tmp/w2a_green_gate_tight.log` (q/k max_abs ≤
  5.3e-4 against the tightened envelope), mutations `/tmp/w2a_mut_m1_conv.log`
  (M1: drop the conv store-rounding → steps 2+ fail), `/tmp/w2a_mut_m2_gdn.log`
  (M2: drop the GDN commit rounding → bit-exactness fails), M5 redo
  `/tmp/w2a_mut_m5_gate_redo.log`; each red, each restored byte-for-byte.
- Allow-list: `Qwen3_5ForConditionalGeneration` added to
  `TenstorrentPlatform::supports_model_architecture` LAST in implementation
  order (the discrete device's by-name refusal is the safety property).

**The L1 finding and the split-shadow fix.** The first e2e bootstrap
(`/tmp/w2a_e2e_bootstrap.log`) threw in `ttnn::prim::IndexedFillDeviceOperation`:
"Statically allocated dataflow buffers on core range [0-0 - 10-9] grow to
2208704 B which is beyond max L1 size of 1572864 B". Root cause:
`ScatterRowsExact` staged TWO FULL PAGES OF THE LAST DIM through indexed_fill's
generic interleaved path (`indexed_fill_program_factory.cpp`: page_size =
padded_last_dim × elem_size, data DFB num_entries = 2); its comment assumed
cols = 32768 (256 KB staging), but the Qwen3.5 GDN ssm_state row is
Hv·Dk·Dv = 16·128·128 = 262144 f32 elems → 2 × 1 MB > the 1.5 MB budget.

The first repair attempt (view `[rows, cols]` metadata-only as `[rows*nb, cb]`)
is REFUTED and recorded here: `tt::tt_metal::view` launches no program even
when the last dim changes (`tensor_ops.cpp` recreates the MeshBuffer over the
same address), BUT device pages are the interleave unit
(`buffer.cpp Buffer::page_address`: bank_offset = aligned_page_size ·
(page_index / num_banks)) — a last-dim-changing view re-maps every page index
to a different bank and scrambles flat order. Measured: one block landed,
three came back scrambled (`/tmp/w2a_diag.log` era). The landed design makes
the SHADOW BORN SPLIT instead:

- `SplitFactor(cols, esz)` picks the smallest divisor F of cols with
  2·(cols/F)·esz ≤ 1 MiB of L1 staging (F=1 → byte-identical legacy form;
  F=2 at cols=262144 → blk=131072, 512 KB pages, 1 MB DFB — inside budget).
- `EnsureGdnCacheDevice` uploads the SAME flat host bytes as
  `[rows·F, cols/F]` (pure geometry at H2D time — no extra copy); logical dims
  stay in the slot bookkeeping, so downloads and volume checks are unchanged.
- `ScatterRowsExact` takes `factor`, expands each live slot s into block ids
  s·F+j, expands the NULL-compaction row list the same way, and views rank-4
  with the last dim UNCHANGED (the safe metadata-only case); still ONE launch.
  Per-(slot, block) last-of-duplicates wins because entry order survives
  inside every block group.
- `GdnDecodeKernel` gathers through a factor-adapted one-hot
  (`UploadOneHot(..., F)`), reshapes S_new to `[B·F, blk]` (one exact
  per-tile-CB regroup when F>1 — the only added device movement), applies the
  bf16 storage rounding, then scatters. `GdnStateGatherKernel` expands its
  index rows and `has_initial_state` mask to block-rows.
- The `CausalConv1dUpdate` scratch-row indexed_fill is NOT in this hazard
  class — computed, not guessed: its tensors are TILE, where indexed_fill
  page = tile size (4 KB), not row width; its `ttnn::reshape`s go through
  reshape_tiled with per-tile CBs. Unchanged.

Focused tests (`tests/vt/test_tenstorrent_backend.cpp`): a wide-row
gather/scatter arm at cols = 262144 (bit-exact vs CPU oracle, untouched slot
kept) and a wide-state decode arm with a NULL row (compaction feeding the
split launch, out AND cache within the standing envelope).

Mutation evidence for the fix:

- M-A (chunking removed — `SplitFactor` pinned to 1): the wide arms throw the
  EXACT bootstrap throw ("grow to 2208704 B ... beyond max L1 size of
  1572864 B"), test case FAILED → `/tmp/w2a_red_l1_wide.log`. Restored → green
  `/tmp/w2a_green_l1_wide.log` lineage.
- M-B (bid expansion drops the block offset — `s*F` instead of `s*F+j`): wrong
  columns written, wide-arm cache compare fails → `/tmp/w2a_mut_bid_offset.log`
  (1058/1059, Status FAILURE). Restored → green `/tmp/w2a_restore_check.log`.

Focused suites green after the fix: gather/scatter + decode (116 assertions),
round-trip (71), conv-update + prefill — `/tmp/w2a_focused_green.log`,
`/tmp/w2a_focused_green2.log`, `/tmp/w2a_focused_green3.log`.

Known upstream instability surfaced by the new width (recorded, not waived):
with SEVERAL consecutive wide-cache arms in one process, tt-metal's
RealtimeProfilerManager receiver thread can abort mid-test with host heap
corruption ("free(): invalid size", gdb lands in
`D2HSocket::pages_available`) — an upstream D2H-socket fragility in the same
class as the recorded #1486 teardown noise, reachable only because W2a makes
262144-wide shadows runnable at all. The landed focused arms use the minimal
shape that stays stable across reruns; if a future gate hits it, rerun and
record, do not widen the tolerance or skip the arm.

### W2b — e2e bootstrap: the DevicePool tenancy defect (fix in flight)

The first W2b bootstrap died at a readback guard:
`unexpected result size: got 2048 want 4096 out_shape=4x1024` from
`DownloadToHost` (`/tmp/w2b_diag_run1.log`, after runs 2-5 narrowed it — run 5
is `/tmp/w2a_e2e_bootstrap_run5.log`). The slot-trace instrumentation
(`VT_TT_SLOT_TRACE=1`, `/tmp/w2b_diag_run5.log`) pinned it:

- `RmsNormKernel -> EnsureDevice2D -> EnsureHost` reads the layer input
  `[T,1024]` f32 and finds the slot still holding the PREVIOUS tenant's device
  shadow (`[8,256]` bf16 — the full-attn `k_out` geometry of an earlier step).
- The buffer was registered ONCE and never freed or re-registered: it is a
  `DevicePool` block. `DBuf` draws scratch from the pool; a pool HIT returns a
  retained block WITHOUT calling `Backend::Alloc`, so nothing ever told the TT
  backend that the block changed tenants. The slot kept the dead tensor's
  shadow (`host_current=false`) and the next reader downloaded it.
- This is why the row (and Mistral/GDN before it) never saw it: their paths do
  not recycle pooled scratch into read-first tensors on TT; Qwen3/Qwen3.5's
  `DBuf` pool does. It is ALSO silent-corruption shaped: when the stale
  geometry happens to match, the new tenant inherits the old bytes marked as
  current.

Fix (three pieces, in this change):

1. `vt::Backend::OnScratchBlockAcquired(void*)` — new defaulted virtual;
   `DevicePool::Get` calls it on every free-list hit. Default no-op: no other
   backend keys residency by pointer.
2. `TenstorrentBackend` overrides it with `MarkHostWritten(p)` — the same state
   a fresh Alloc registers.
3. `CommitDeviceLogical2D` now asserts the committed device tensor's volume
   against `rows*cols`, so a producer bug dies at its producer instead of at a
   later reader.

Red evidence for the fix is the bootstrap failure itself (runs above). The
mutation (pool hit stops notifying) must rethrow the same mismatch — recorded
below once run.

### W2b — e2e bootstrap GREEN, then the output-quality bisect

With the tenancy fix in, the bootstrap completes: `Status: SUCCESS!`, 104/104
assertions, ids dumped to `our_ids_tenstorrent.i32` (16x16), BACKEND PROOF
`kPagedAttention=1536 kGdnDecode=4320 kCausalConv1dUpdate=4320, 0 declines`
(`/tmp/w2b_fix_run1.log`; exit 139 is the #1486 teardown). The teacher-forced
golden pair was derived (`qwen3-neartie-gap-transformers.py`, torch
2.7.1+cpu / transformers 5.8.1 in `/home/lu_zero/Sources/tt/venv-qwen35gap`)
— and the DUMPED SEQUENCE IS DEGENERATE: every prompt collapses to
`,`/space (`11`/`220`) within a step or two. transformers teacher-forcing
AGREES with our tokens per step (`tf_argmax == our` throughout), so the gap
golden faithfully describes what our engine emitted; the engine's emission is
what is wrong.

Bisect (all on one prompt, `vllm-cli --device auto --max-tokens 12`,
~4 min/run after the first JIT warm run):

- **CPU arm clean**: `--device cpu` prints ` Paris.` — identical to the ROCm
  golden. Model/metadata plumbing is correct; the divergence is TT-resident.
  (First CLI attempt used a STALE pre-W2a binary from 12:47 and replayed the
  already-fixed L1 throw — void run, rebuilt before any conclusion.)
- **bf16-state arms exonerated**: `VT_GDN_STATE_BF16=0` collapses identically.
- **Host-free decode is the carrier**: `VT_TT_HOST_FREE_DECODE=0` prints ` the
  capital of the United States.` — coherent greedy text. Ambient (default ON)
  collapses.

This matches the spec's own risk line ("host-free decode interplay with the
new elementwise ops"). MECHANISM CORRECTED after reading #1625: on TT,
trace capture is OPT-IN (`VT_TT_DECODE_CAPTURE`; the multi-request hang made
`support_static_graph_mode()` decline by default), so the AMBIENT leg here is
HOST-FREE EAGER decode, not capture. The carrier statement stands
(`=0` — the legacy host-staged decode — is coherent while ambient collapses);
what breaks is the host-free eager chaining of the NEW ops' per-step inputs
and device-resident outputs (the dense ops each carry explicit host-free
plumbing — DecodeIdsCache, RacIdxCache warm hooks, rope warm-before-capture;
none exists for kGdnPostConv/kSigmoidGateBf16/kAttnQkNormRopeGate or the GDN
state path). The eager op-level suites (36/36) cannot see it; the dump-only
bootstrap had no correctness bar; only the `=0` leg runs the ops uncaptured
by host-free. Logs:
`/tmp/w2b_tt_ambient2.log` (collapse), `/tmp/w2b_tt_f32state.log` (collapse),
`/tmp/w2b_tt_nohostfree.log` (coherent).

### W2b — the =0 gate red is a SEQUENCING error, and it sharpened the diagnosis

The first golden pair was derived from the DEGENERATE bootstrap sequence
(before any output-sanity check) — a process mistake, recorded here so it does
not repeat: validate emitted text BEFORE deriving goldens. The pair was
quarantined to `/tmp/w2b_quarantined_goldens/` (never committed).

The `=0` full-gate run against that pair then failed at the anchor REQUIRE —
correctly, since the committed anchor described the broken engine:
`prompt[0] tok=0 engine=279 committed anchor=11`
(`/tmp/w2b_gate_nohostfree.log`). Decoded: our `=0` leg emits `" the"` where
the ROCm oracle (and OUR CPU ARM, verbatim) emit `" Paris"`, and the broken
ambient leg emitted `","`. Three-way split at token 0:

| leg | tok0 | character |
|---|---|---|
| ROCm oracle + our CPU arm | ` Paris` (11751) | reference |
| TT `VT_TT_HOST_FREE_DECODE=0` | ` the` (279) | fluent; diverges from both oracles |
| TT ambient (host-free eager) | `,` (11) | degenerate collapse |

So there are TWO open defects, not one:

1. **Ambient collapse** (host-free eager). The host-free branches are NOT
   decode-only — `CopyDeviceDeviceIfCapture`, `MemsetDeviceIfCapture`,
   `PreferDeviceRope`, and the forced-device residual+RMS all activate during
   PREFILL too, and ambient's TOKEN 0 is already wrong. Next probe (board
   freeing): `vllm-cli --max-tokens 1` ambient vs `=0` — pure-prefill
   divergence convicts the prefill-active host-free branches without decode
   in the picture; then `VT_TT_TRACE_DEBUG=1` for the tensor flow.
2. **`=0` numeric divergence** (" the" vs " Paris"). Fluent text but a real
   first-token divergence from BOTH oracles. Either an honest near-tie (the
   teacher-forced gap band decides) or a second milder numeric defect in the
   eager path. The `=0` re-bootstrap WITH `VT_DUMP_IDS=1`
   (`/tmp/w2b_bootstrap_nohostfree.log`) captures this leg's true sequence;
   teacher-forcing it answers near-tie-vs-defect directly.

### W2b — the =0 leg is near-tie-clean EXCEPT 4 steps; two distinct defects

The `=0` bootstrap completed green (`/tmp/w2b_bootstrap_nohostfree.log`,
Status SUCCESS, ids dumped) and its sequence is HEALTHY: 127/256 tokens
identical to the ROCm oracle, fluent throughout (p4 emits correct fibonacci
code; p15 tracks the oracle's structure). Teacher-forced gaps
(`qwen3-neartie-gap-transformers.py`, committed pair
`our_ids_tenstorrent.npy` + `neartie_gap_mnats_tenstorrent.npy`):

- **252/256 steps inside the 500 mnats band**, most at exactly 0.
- p0 tok0 IS a genuine near-tie: our `" the"` vs transformers' `" Paris"` at
  **375 mnats** — inside the band; not a defect.
- FOUR steps over: p1 tok0 (**6813**), p1 tok8 (4688), p10 tok14 (1000),
  p15 tok13 (625). Real divergences, localized — NOT a global math error.

So the row owes TWO fixes, in this order:

- **2a (primary, blocks the default path): ambient host-free collapse.**
  Prefill-active suspicion recorded above; trace probe queued.
- **2b (blocks the =0 leg): 4 out-of-band steps.** Localized numeric
  divergence under the eager path; bisect by prompt/layer after 2a, since the
  full gate needs both legs anyway.

### W2b — DEFECT 2A ROOT-CAUSED AND FIXED: the gemma `+1` dropped on the device arm

The bisection chain, each step board-pinned:

1. **Prefill, not decode.** `vllm-cli --max-tokens 1` (pure prefill, zero
   decode steps): ambient `,` vs `=0` ` the` (`/tmp/w2b_prefill_{AMB,NOHF}.log`).
   The host-free branches active during PREFILL are the suspects.
2. **Kill-switch bisect** (scratch `VT_TT_HF_DISABLE`, since removed): with
   ALL FOUR prefill-active branches declined — device rope
   (`PreferDeviceRope`), forced-device residual+RMS, d2d `Copy` hook, device
   `Memset` — ambient emits ` the` (`/tmp/w2b_hfdisable.log`). Per-branch
   split (`/tmp/w2b_split_*.log`): **only `residual` convicts** (armed → `,
`; declined → ` the`); rope/copy/memset are innocent.
3. **Root cause:** `RmsNormKernel`'s device arm hands ttnn::rms_norm the RAW
   affine and has ALWAYS been gemma-host-only — the file said so ("Gemma
   style (w+1) is host-only for now — Qwen3 does not set gemma=true").
   Qwen3-dense never sets gemma; **Qwen3.5 sets gemma=true at 21 RmsNorm
   sites**, so under host-free every forced-device norm silently dropped the
   `+1`. Under `=0` the `args.gemma` clause routes to the host arm, which is
   why that leg was merely near-tie-flavored instead of collapsed.

Fix: when `args.gemma`, bake `w+1` host-side in f32 (the oracle's order, same
treatment as the fused preamble's `weff`) and upload as an F32 gamma — a
TRANSIENT upload that deliberately bypasses the `EnsureAffine1D` slot cache,
because the cached form is the RAW weight and a non-gemma consumer of the
same buffer must never read the baked one.

Evidence ladder:

- GREEN e2e: ambient CLI now emits ` Paris.` — the ROCm oracle's AND the CPU
  arm's exact continuation (`/tmp/w2b_gemma_fix.log`). The fix even improves
  on the pre-fix `=0` leg's `" the"` near-tie.
- Op-level pin added:
  `kTENSTORRENT kRmsNorm gemma matches a host F32 reference (w+1)` in
  `tests/vt/test_tenstorrent_backend.cpp` (small weights make the +1 dominate).
- Mutation RED: dropping the `+1` fails that case
  (`/tmp/w2b_gemma_red.log`, CHECK max_abs_diff < 0.5f NOT correct); restored
  → green `/tmp/w2b_gemma_green2.log`.

### W2b — post-fix state: collapse healed, 2b isolated as leg-independent

The post-fix ambient bootstrap ran against the stale pre-fix goldens still on
disk, so it took the COMPARE path: 3 `prompt_ok` reds vs an obsolete anchor
and a correct final `REQUIRE(fail==0)` failure (`/tmp/w2b_bootstrap_fixed.log`)
— expected, not a regression. Fresh ids were dumped anyway and re-derived into
a new committed-to-worktree pair describing the FIXED engine.

Fresh ambient sequence: 247/256 steps inside the band; out-of-band steps are
p1 tok0 (**6813**), p1 tok10 (3125), p1 tok4 (1250), p1 tok5 (1188),
p10 tok14 (1000), p7 tok11 (812). Prompt 1 carries four of six.

**2b is INTRINSIC, not cross-prompt contamination**: standalone CLI run of
prompt 1 reproduces the identical divergent continuation
(` when the world was a place of wonder...` vs oracle
` in a world where everything was made of atoms...`,
`/tmp/w2b_p1_solo.log`). The engine is self-consistent (solo == batch),
fluent, near-tie on >96% of steps — but computes specific contexts
differently from BOTH oracles by up to 6.8 nats.

W2c queue (next sessions):

1. **2b bisect by layer** on prompt 1: `VT_DUMP_ATTN` covers full-attn layers
   only; GDN layers need an equivalent per-layer dump hook before the
   first-drift layer can be named. Then op-level replay of the drifting
   layer's real inputs vs the CPU arm.
2. Full gates both legs (blocked by 2b only).
3. Fresh review + landing of the whole wave (pool tenancy fix, gemma fix,
   diagnostics, golden pair, USAGE entry).

### W2c — 2b localized to the qkvz projection output

Instrumentation added (env-gated debug hooks): `DumpGdnStage` inside
`GdnBlockPaged` (dumps `mixed`, `conv`, `postconv_q/v`, `core`, `gated` per
invocation under `$VT_DUMP_ACT`) and a pre-layer dual-stream snapshot
(`layer_-1_{hidden,res}`) beside the existing per-layer `VT_DUMP_ACT` loop.

Bisect chain on prompt 1, prefill-only, ambient vs CPU:

1. Residual stream after EVERY layer diverges from layer 0 onward
   (`layer_0.bin`: max_abs 2.12, mean 0.11) — born in the first layer, not
   accumulated.
2. Pre-layer inputs (`layer_-1_*`): BIT-IDENTICAL between arms. The defect
   lives inside layer 0's mixer (a linear-attention/GDN layer).
3. `VT_DUMP_ACT_SUB` stage probes for layer 0: `post_input_norm` max 0.0098
   corr=1.000000 (the gemma-baked device residual+RMS arm is numerically
   sound); `post_attn_norm` max 3.30 corr 0.962. The divergence enters in
   the mixer.
4. First mixer checkpoint — the qkvz PROJECTION OUTPUT (`mixed`,
   [5,6144]): max_abs 2.1, corr 0.9986, with ~92/30720 elements grossly
   wrong including SIGN FLIPS (cpu +1.14 vs tt -0.96), scattered across
   columns and rows. Not a layout block, not accumulation noise.
5. `VT_GDN_IN_BF16=0` (f32 activations both arms): IDENTICAL divergence —
   the bf16 activation path is exonerated; so is the dtype policy.
6. Standalone-vs-batch prompt 1 identical ⇒ not cross-prompt state leak.

**Suspect: the qkvz projection GEMM on TT** (kMatmul / merged-qkvz leaf)
producing scattered grossly-wrong elements at [T,1024]x[1024,6144] with
real layer-0 weights and verified-clean inputs. Next session: op-level
replay of exactly that shape with the dumped `post_input_norm` as input
and the real weight column, compared element-wise against the CPU result;
then read the winning kernel's accumulation/padding path for the defect.

### W2c — replay verdict: the op is clean; the CAPTURES under host-free are not

Op-level replays with the REAL captured bytes, isolated:

- kMatmulBT bf16 x bf16 -> bf16, [5,1024]x[8192,1024] (qkvz): worst err
  0.163 over ±20 values — envelope. CLEAN.
- kMatmulBT bf16 x bf16 -> F32, synthetic: CLEAN (0.047).
- kMatmulBT with the REAL h0 + REAL w_ba bytes, split-arm signature
  ([5,1024]x[16,1024] -> F32): CLEAN (0.045 / 0.057).
- Resident weights captured from BOTH arms byte-identical and clean
  (`w_ba.bin`: no NaNs, ±0.18 range).

And the decisive behavioral test: **`VT_POOL_BYPASS=1` produces output
IDENTICAL to the pooled run** — pool-block state cannot be carrying the
divergence, and (critically) the engine's TEXT IS FLUENT even where the
mm-captures claimed `b/a` outputs were 1e38/NaN. If `b/a` were truly
garbage in the live dataflow, exp()/sigmoid would explode and the text
would collapse. It does not.

**Therefore: the stage-dump capture path itself is UNRELIABLE under
host-free decode.** The DBuf-tmp + Backend::Copy download pattern used by
every probe hook (DumpStage, DumpGdnStage, the mm/qkvz captures) can serve
bytes that do not match what the live chain consumes — most plausibly via
the CopyDeviceDeviceIfCapture clone interaction (clone enqueued on the
device while the readback path resolves different residency state). This
invalidates the specific "projection GEMM garbage" and "ba garbage"
findings above as statements about live data, and with them the
layer-localization derived from those dumps.

What SURVIVES (behavioral ground truth, independent of captures):

- Post-gemma-fix engine: fluent, deterministic, 247/256 steps inside the
  near-tie band; six out-of-band steps concentrated on prompt 1 (worst
  6.8 nats at tok 0); solo == batch; f32-input leg reproduces it;
  VT_POOL_BYPASS neutral.

W2c revised plan:

1. Build a TRUSTED capture: either take dumps on the `=0` leg (no host-free
   hooks active) and trust arm-symmetric differences, or add a
   dual-read verification to the probe (download twice via independent
   paths + Synchronize, flag mismatches) before trusting any single dump.
2. Re-run the layer/stage localization UNDER THAT PATH. The defect is real
   (six band violations, oracle-referenced) but every named-kernel claim
   from unreliable dumps is withdrawn.

The pool-tenancy fix is ORTHOGONAL to both and already proven necessary:
without it no run reaches a summary (runs 2-5 crashed). Its mutation cell
(same bootstrap config, guard neutered) is in flight — expected to rethrow
the stale-shadow mismatch, closing the red/green pair with
`/tmp/w2b_fix_run1.log`.

**Mutation cell CLOSED**: guard neutered + same bootstrap config rethrows the
EXACT original signature (`unexpected result size: got 2048 want 4096
out_shape=4x1024 ctx=EnsureHost dev[8x256 dt=1]`, Status FAILURE —
`/tmp/w2b_mut_pool_gate2.log`); restored byte-for-byte → green lineage
`/tmp/w2b_fix_run1.log`. The fix's guarantee ("a pooled block never inherits
its previous tenant's residency") is mutation-pinned.
