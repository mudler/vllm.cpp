# ROCm `gfx1151` Qwen3.8-27B Q4_K decode evidence, 2026-09-01

Immutable evidence index for the first quant-matched decode measurement on
`strix:gpu0` (Radeon 8060S, Strix Halo), against the pinned llama.cpp oracle
built for the same architecture in the same lease.
Issue [#2497](https://github.com/mudler/vllm.cpp/issues/2497), row
`BACKEND-ROCM`.

## Disposition

**The speed axis on this arm is INADMISSIBLE, and the reliability finding is
the result.**

This run's throughput figures are recorded below for completeness and are
quotable as nothing. `AGENTS.md` §Gates requires the declared token-exact gate
before a performance result is accepted, and this arm does not have one: see
the correctness precondition immediately below. That precondition was not
checked before this run was taken, which is a defect in this measurement and
not a caveat on it.

What the run does establish, and what stands on its own:

- **Our arm hangs the GPU on 2 of 3 legs** on the plain Q4_K GGUF target, with
  zero CPU reference-tier ops and no DFlash2, while llama.cpp completes 3 of 3
  on the identical bytes in the same lease. That is a reliability finding, not
  a speed claim, and nothing about the token gate weakens it.
  [#2511](https://github.com/mudler/vllm.cpp/issues/2511) owns it.
- **Our GGUF reader cannot load the `UD` quant family** over one missing ggml
  type. [#2510](https://github.com/mudler/vllm.cpp/issues/2510) owns it.
- **The rig reproduces the published llama.cpp band on this silicon.** That is
  a single-engine measurement of the oracle and carries no cross-engine ratio,
  so it is unaffected.

## The correctness precondition this run did not meet

`docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md` records
`TOKEN_GATE=FAIL` for **this exact artifact** — `gguf_sha256`
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`,
`gguf_size` 17,106,775,008, against the same `b10451` oracle. The tokenizer is
exact on 6 of 6 prompts; the generation diverges on 5 of 6. That document
states in its own words: "**No speed axis was run and none may be quoted**",
and "No memory axis is accepted either, for the same reason."

The divergences are rank-2 losses under 0.18 logits, with 282 of 288 steps at
rank 1 — a precision difference in the quantized compute path rather than a
wiring defect. That makes the arm *close*, and close is not the standard. The
ratified distributional band does not rescue it either: `AGENTS.md` admits that
band only where the oracle's greedy decode is non-deterministic, and this
oracle's is deterministic, so the band's premise fails and was explicitly not
reached for.

So the correct reading of the numbers below is that they rank two engines that
are not yet computing the same thing. A speed comparison becomes admissible on
this arm when its token gate passes, and not before.

The deficit sits in the band already measured on `gfx1200` (2.40x dense in
[#1863](https://github.com/mudler/vllm.cpp/issues/1863), 4.25x MoE in
[#1400](https://github.com/mudler/vllm.cpp/issues/1400)), so the ROCm decode
deficit is not specific to Strix Halo and this board adds a third point to it.

## What this is not

**Not a gate.** No in-tree harness samples AMD clock state — that is owed by
[#2381](https://github.com/mudler/vllm.cpp/issues/2381) — so every clock figure
here comes from an ad-hoc sysfs sampler carried beside this run and is recorded
as such. `tools/bench/gpu_clock_state.py`, which `.agents/benchmarking.md`
names as the one helper, reads NVIDIA fields and does not run on this board.

**Not a vLLM comparison.** vLLM is the bar, and it does not run on `gfx1151`.
llama.cpp appears here as the explicitly labelled secondary comparison
`.agents/benchmarking.md` permits, and it is the GGUF k-quant floor its oracle
file scopes it to.

**Not the DFlash2 / MTP arm.** That arm reaches CPU reference-tier ops and
hangs this GPU; [#2377](https://github.com/mudler/vllm.cpp/issues/2377) owns it
and it is out of scope here. The published community figures this run is placed
against are mostly *speculative-decoding* results; only their spec-off baselines
are comparable to anything measured here.

## Environment

| | |
|---|---|
| Device | `strix:gpu0`, Radeon 8060S Graphics, `gfx1151` (0x1151) |
| VRAM | 65536 MiB firmware carve, 59934 MiB free before load; Wave Size 32; VMM no |
| Host RAM | 62 GiB visible to the OS, 32 CPUs |
| Kernel | 6.14.0-36-generic |
| ROCm | 7.2.4 |
| HIP compiler | AMD clang 22.0.0git, roc-7.2.4, `f58b06dce1f9c15707c5f808fd002e18c2accf7e` |
| Boot id | `a5bc8128-f6ad-4767-8614-6923f88032e1` |
| Lease | every leg inside an `rc` lease; no `ssh`, no file mutex, one job at a time |

The GPU's memory is carved from system RAM by firmware, so it has a hard
ceiling like a discrete card and the split is a BIOS setting rather than a
driver flag.

## Identities

| Artifact | Identity |
|---|---|
| vllm.cpp | `11fed3ba56b8f823c07032416982a44a8c0967b5` (`origin/main`), bundle sha256 `abcb1e84219ce13f00c363a6441ae3e5f9bdfb10f5e14afadd0131a51fccbace` |
| vllm.cpp build flags | Release, `VLLM_CPP_HIP=ON`, `VLLM_CPP_HIP_ARCHITECTURES=gfx1151`, CUDA OFF, Triton OFF, `-j 4` |
| `vllm-cli` binary | sha256 `a703b83dd8954ba6dd3cbe82efcd38083c1d55492bbbaecf5c406f7c6efd646f` |
| llama.cpp | tag `b10451` = `10bf611e533d81f739128304991c5e133c6aebd8`, the pin in `.agents/oracles/llama-cpp.md` |
| llama.cpp source transfer | `git archive` tarball sha256 `9f92a61ddd1bc65d03db32203f8c8bbc9adf36c227eed804fcf60d811054fb72` |
| `llama-bench` binary | sha256 `e2acbe26f4ef214ca5b3658a1062929242345a1205f9a13d6af841bfa6c3f2c1` |
| `llama-cli` binary | sha256 `d563d9877adb49b4ae2c0a6c23e4019548ffc0c7e2d3a7c4339af3f636646a55` |
| llama.cpp build flags | Release, `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1151`, rocWMMA FATTN OFF |

**One identity weakness, recorded rather than hidden.** The llama.cpp source
reached the box as a `git archive` tarball, because the local checkout is
shallow and `git bundle` cannot carry a shallow history. The tarball has no
`.git`, so the built binary self-reports `version: 0.1.0-dev (build 0, commit
unknown)` and cannot attest its own pin. The pin is instead attested by the
tarball sha256 above, which was produced from the tag object
`10bf611e533d81f7…` on the reading machine, and by the binary sha256.

## Artifacts

| File | Bytes | sha256 | Loadable by us |
|---|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` | 17,106,775,008 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | yes |
| `Qwen3.8-27B-UD-Q4_K_M.gguf` @ `4ca720788d1e01f1bff70c033e0d0028fd02e502` | 16,464,440,224 | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` | **no** ([#2510](https://github.com/mudler/vllm.cpp/issues/2510)) |

Both from `unsloth/Qwen3.8-27B-GGUF`. Every sha256 above was verified on the
worker against the staged bytes before any timing ran.

**Why two artifacts.** The measurement wanted the file the published community
rows use, which is a `UD` dynamic quant. Our GGUF reader has no case for ggml
type id 21 (`IQ3_S`), and `UD-Q4_K_M` carries four such tensors out of 866
(`UD-Q4_K_XL` carries exactly one), so the loader refuses the whole file. The
plain `Q4_K_M` is the artifact the `QUANT-QWEN38-27B-GGUF-ARM` row was
header-verified against and it carries no `IQ3_S`. The `UD` file is therefore
retained here only as the denominator-side control and as #2510's reproduction.

The published `UD-Q4_K_XL` figures cannot be reproduced byte-for-byte in any
case: that name served 17,923,394,624 bytes when those rows were measured and
serves 17,559,178,144 bytes today, re-quantized in place under an unchanged
name. This is the reason the denominator here is measured rather than quoted.

## Method

- Both arms in one lease, one job at a time, on a box no other job shared.
- Each artifact copied from CIFS `/workspace` to worker-local `/tmp` before any
  load, and its sha256 verified there.
- Order-alternated legs across three rounds (ours first on odd rounds, the
  oracle first on even), so a monotonic drift cannot land on one arm.
- Our arm loads once and runs four completions (`--repeat 4`); run 1 is the
  in-process cold run and is discarded for that named cause. The oracle is
  `llama-bench -r 3`, which does its own warmup.
- Identical prompt, 64 output tokens, batch 1, greedy, on both sides.
- `VT_OP_PROVIDER_STATS=1` on our arm, asserted to zero CPU reference-tier
  hits. `gfx1151` is integrated, so `docs/ROCM.md`'s reference tier is
  reachable and a silent fallback would otherwise be invisible in throughput.
- A clock sample every 250 ms per leg, written to worker-local disk rather than
  CIFS, because a 4 Hz flush against the share stalls the sampler and distorts
  the very sample spacing the window is judged on.

## The layer-count term, settled

`.agents/oracles/llama-cpp.md` warns that at `b10451` llama.cpp loads 851 of
this artifact's 866 tensors and ignores all 15 of `blk.64`, four of which are
the `nextn.*` MTP head, so "an arm that runs block 64 does strictly more per
token than this denominator". That warning is about *running* block 64, and on
this pairing neither arm does.

Our loader does not spend `block_count` on the trunk. `RefuseUnaccountedQwen3_5Gguf`
(`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:1735`) accounts the
file against `num_hidden_layers` trunk layers **plus** `DeclaredMtpDepth(config)`
MTP blocks as separate populations, and refuses rather than silently dropping a
tensor nothing reads. So the trunk is 64 blocks on both sides, and `blk.64` is
the drafter, which a non-speculative decode never enters.

The residual difference is therefore **residency, not compute**: we load and
hold `blk.64`'s tensors while llama.cpp declines them, roughly 290 MB on this
file. That belongs in the memory column, not in the throughput ratio. The
per-token work is 64 trunk blocks on both arms, which is what makes the ratio
below a performance comparison rather than a configuration one.

## Harness control

Before interpreting any number of ours, the pinned oracle was run on the `UD`
file the published rows use, to check whether this rig reproduces their
ballpark at all.

| | tok/s | 
|---|---|
| llama.cpp `b10451` HIP, `UD-Q4_K_M`, `-ngl 99`, `n_gen 64`, `r 3` | **12.616 ± 0.026** |
| Published Linux ROCm/HIP `gfx1151`, `UD-Q4_K_XL`, 96K ctx, f16 KV | 10.5–11.1 |
| Published Linux Vulkan/RADV `gfx1151`, `UD-Q4_K_XL`, 131K ctx, q4_0 KV | 11.9 |
| Published Windows Vulkan `gfx1151`, `UD-Q4_K_XL`, 32K ctx, q8_0 KV | 11.5 |

Published figures from https://github.com/sudoingX/qwen38-mtp, spec-off arms
only. The measured 12.616 sits slightly above them, which is the expected
direction: `UD-Q4_K_M` is 6.2% fewer bytes than `UD-Q4_K_XL`, and `llama-bench`
holds a minimal context against their 32K–131K. The rig therefore agrees with
the community's numbers rather than merely citing them.

## Results, recorded for completeness and quotable as nothing

Read the correctness precondition above before reading this section. These
figures do not establish a floor, a ratio of record, or a gap that may be cited
elsewhere. They are kept because the run happened and deleting evidence to
reduce context is forbidden.

Prompt `The capital of France is`, 64 output tokens, batch 1, greedy, `-ngl 99`
against 64 trunk blocks resident, three order-alternated rounds.

| Leg | Result | tok/s |
|---|---|---|
| `vllmcpp-r1` | **rc=139, GPU Hang** | — |
| `vllmcpp-r2` | rc=0 | cold 2.837, then **4.523 / 4.382 / 4.510** |
| `vllmcpp-r3` | **rc=139, GPU Hang** | — |
| `llamacpp-r1` | rc=0 | 12.239 ± 0.048 |
| `llamacpp-r2` | rc=0 | 12.242 ± 0.058 |
| `llamacpp-r3` | rc=0 | 12.249 ± 0.043 |

| | tok/s | |
|---|---|---|
| vllm.cpp ROCm, warm median of 3 | **4.510** | spread 4.382–4.523, 3.2% |
| llama.cpp `b10451` HIP, median of 3 legs | **12.242** | legs agree to 0.08% |
| Ratio | 2.71x behind, **INADMISSIBLE** | |

Run 1 of the completing leg (2.837 tok/s) is the in-process cold run and is
discarded for that named cause, not for being inconvenient; the three warm runs
that follow it agree to 3.2%.

`VT_OP_PROVIDER_STATS=1` reported **zero** CPU reference-tier notices on every
one of our legs, the hung ones included. Every op resolved `selected=vt-native`.
This deficit is therefore native ROCm kernels, not a silent host fallback.

**The hang is intermittent, and that is the finding.** Two of three legs died
with `HW Exception by GPU node-1 ... reason :GPU Hang` after roughly 65 s, on
the plain GGUF target, with no DFlash2 and no reference-tier op involved. In
the same lease, on the identical file, llama.cpp completed 3 of 3.
[#2377](https://github.com/mudler/vllm.cpp/issues/2377) recorded this signature
against the DFlash2 arm and named the CPU reference tier and an un-overridden
`RocmBackend::FlushPending()` as the first root-cause candidate. Neither is
present here, so that candidate cannot be the whole cause. The hang on this
plain path is owned by
[#2511](https://github.com/mudler/vllm.cpp/issues/2511).

## Memory and context

`--fit` resolved no placement ("the model has no routed-expert layers, so a
placement has nothing to move"), and auto-fit reduced `max_model_len` from
262,144 to **8,192** to fit a 256-block x 32-token KV cache. So our arm holds a
far smaller context than any published row (32K–131K) and than llama.cpp's own
default here. A larger KV would not make our arm faster, but the configurations
are not equal on this axis and the comparison should not be read as if they
were.

## Clock

Sampled at 4 Hz from `/sys/class/drm/card*/device/{gpu_busy_percent,pp_dpm_sclk}`
per leg. Figures below are the >=90%-busy compute window; see the note beneath.

| Leg | samples | compute samples | median MHz | mean MHz | spread |
|---|---|---|---|---|---|
| `vllmcpp-r2` | 375 | 210 | 2862 | **2851.1** | 8.2% |
| `llamacpp-r1` | 98 | 56 | 2663 | 2683.0 | 9.5% |
| `llamacpp-r2` | 98 | 56 | 2640 | 2678.2 | 10.2% |
| `llamacpp-r3` | 118 | 56 | 2668 | 2692.5 | 10.7% |
| harness control (UD file) | 91 | 54 | 2724 | 2744.1 | 12.0% |

**The cross-arm offset points the safe way.** Our arm held a 6.2% *higher* mean
clock than the oracle (2851.1 against 2684.6) and was still 2.71x slower.
Normalising our throughput to the oracle's clock gives 4.247 tok/s and a
**2.88x** ratio, so the offset understates the deficit rather than creating it.
The measured 2.71x is therefore a floor on the gap, not an artifact of it.

The offset itself is largely an artifact of asymmetric windows: `vllm-cli`
prints `generate_start_unix` / `generate_end_unix` per run, so our window is
clipped to generation exactly, while `llama-bench` emits no such stamps and its
window still carries load and setup even after the busy filter. That asymmetry
is worth removing before any ROCm ratio is quoted as a gate.

**A rule that does not transfer, recorded for [#2381](https://github.com/mudler/vllm.cpp/issues/2381).**
`.agents/benchmarking.md` requires the SM-clock spread within a run to stay at
or below 5%. On the harness-control leg the spread is 64.5% over every busy
sample and 12.0% over the >=90%-busy compute window (2550–2878 MHz, median
2724). That rule was calibrated on `dgx.casa`, a datacenter part with
persistence mode and a dedicated power budget. This APU shares its power and
thermal budget with 32 CPU cores on one package, so a wider sustained-load
swing is a property of the hardware rather than evidence of a dirty
measurement. An AMD clock rule needs its own calibration; it cannot inherit the
NVIDIA number. What still transfers unchanged is the cross-arm requirement:
the two arms' clock medians and means must agree, because that offset is the
term that lands in the ratio.
