# `qwen38-27b-q4km-gfx1151` — Qwen3.8-27B Q4_K_M on Strix Halo, three engines

llama.cpp, the pinned vLLM and vllm.cpp were run on one board, on one artifact,
on one prompt. The first lease, on 2026-09-04, interleaved all three at one
token count and lost our own arm to a hang. A second lease, on 2026-09-05, ran
our arm alone at two token counts, on a head whose identity was asserted rather
than printed. This file has what each engine did, and which lease it did it in.

## Disposition

Mixed, and the mixture is the point.

**All three engines now have a figure. Ours is the slowest of the three, and it
is not a gated result.** In lease 1 vllm.cpp faulted the board on every leg.
That fault was the pre-#2511 binary reproducing rather than the engine failing:
the harness recorded our revision and never asserted it
([#2933](https://github.com/mudler/vllm.cpp/issues/2933)). Re-measured in lease
2 on a head asserted five ways, our arm completed 8 legs of 8 with no fault.

## Correctness comes first, and it is not passing

**This is a survey. No ratio on this page is a gated result.** It is published
under a recorded developer decision of 2026-09-04, and the correctness state is
part of the table rather than a footnote:

| comparison | divergent prompts |
|---|---|
| `TOKEN_GATE` | **FAIL** |
| vllm.cpp vs llama.cpp `b10451` | 3 of 6 |
| vllm.cpp vs vLLM (`torch.compile`) | 5 of 6 |
| vLLM (`torch.compile`) vs llama.cpp `b10451` | 3 of 6 |
| vLLM (eager) vs llama.cpp `b10451` | 4 of 6 |

Every divergence is a near-tie at about 0.125 nats, one bf16 ULP.

**No deterministic denominator exists on this path.** llama.cpp's greedy decode
is not stable across its own kernel paths, and vLLM disagrees with itself across
`enforce_eager` on both 27B models tried. Limb 3 of the ratified methodology is
unsatisfied and may be unsatisfiable here.

These four counts are **constants carried from earlier evidence**, and
**neither lease re-measured any of them**. Both fold scripts write them in as
literals; lease 2's `RESULT.json` says so in its own `token_gate.note`. They
describe the same board and the same artifact, and no leg of either run
re-derived them.

## Subject

| | |
|---|---|
| artifact | `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 B, sha256 `7e78da5d…fe169` |
| repo | `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` |
| device | AMD Radeon 8060S, `gfx1151`, ROCm 7.2.4, 64 GiB, `strix:gpu0` |
| lease 1, all three engines | `rc` job `b51afb48-ddb6-438a-b30c-acf46980e918`, 2026-09-04, boot `a5bc8128…f88032e1` |
| lease 2, vllm.cpp only | `rc` job `5c3d309a-d505-4bd4-a173-e9f46f9cf7e0`, 2026-09-05, boot `a5bc8128…f88032e1` |
| prompt | `The capital of France is`, 5 tokens, greedy, `--temperature 0` |
| generation | 64 tokens; lease 2 adds 128 tokens, so that a slope can be taken |

The artifact's sha256 was recomputed on the worker before any timing ran, in
both leases, and reads `7e78da5d…fe169` in both. The two leases carry the same
`boot_id`, so the board was not rebooted between them.

## Method

Both leases run four rounds, discard run 1 of each in-process arm as cold, and
sample the board clock at 4 Hz beside every leg to worker-local disk.

**Lease 1** ran all three engines in every round, with the **engine order**
rotating each round, so drift along the session cannot be read as a difference
between engines.

**Lease 2** ran vllm.cpp alone, twice per round, at 64 and at 128 tokens, with
the **token-count order** rotating each round, for the same reason.

```sh
# llama.cpp, pure decode                                          (lease 1)
llama-bench -m <gguf> -p 0 -n 64 -ngl 99 -r 3 -o json

# vLLM, production configuration (NOT --enforce-eager)            (lease 1)
LLM(model=<gguf>, quantization="gguf", gpu_memory_utilization=0.75,
    max_model_len=4096, enforce_eager=False)

# vllm.cpp                                              (lease 1, and lease 2
#                                                        again with 128)
vllm-cli --model <gguf> --prompt 'The capital of France is' --max-tokens 64 \
  --temperature 0 --repeat 4 --max-num-seqs 1
```

## Results

| engine | legs | fault rate | figure | spread |
|---|---|---|---|---|
| llama.cpp `b10451` | 4 of 4 | 0 of 4 | **12.219 tok/s** decode | 0.133% |
| vLLM `0.26.0.dev0+g5559679229` | 4 of 4 | 0 of 4 | **6.734 tok/s** whole completion | 0.192% |
| vLLM, decode only (derived) | 4 of 4 | 0 of 4 | **11.056 tok/s** | 0.835% |
| **vllm.cpp `c796fea41`**, lease 2 | **8 of 8** | **0 of 8** | **5.305 tok/s** whole completion, 64 tok | 0.829% |
| vllm.cpp, decode only (derived), lease 2 | 4 of 4 pairs | — | **5.397 tok/s** derived | 0.564% |
| vllm.cpp `11fed3ba5`, lease 1 | 0 of 4 | 4 of 4, `rc=139` | none — FAULTED, superseded | — |
| llama.cpp `llama-cli` control | 0 of 4 | 0 of 4 | none — harness defect | — |

### The ratio depends on which definition you take

`llama-bench -p 0` excludes the prompt and every per-request cost. The vLLM leg
times a whole `generate()` call. Those are two different quantities, and the
difference between them is not small here:

| basis | llama.cpp / vLLM |
|---|---|
| whole completion vs pure decode, **mismatched** | 1.814x |
| **decode against decode** | **1.105x** |

**The 1.814x figure compares two different things and should not be quoted.**
The survey's own fold emits it because the leg that would have supplied a
matched llama.cpp row timed out (see below). Read the 1.105x.

### How the decode-only vLLM figure was derived

It is a derivation from two token counts, not a direct reading, and is labelled
so wherever it appears. Each leg timed 64-token and 128-token completions of the
same prompt. The slope between them removes whatever fixed cost each call pays:

| leg | t(64) s | t(128) s | slope tok/s | 1-token call s |
|---|---|---|---|---|
| r1 | 9.5094 | 15.3057 | 11.041 | 3.860 |
| r2 | 9.4913 | 15.2916 | 11.034 | 3.848 |
| r3 | 9.5018 | 15.2541 | 11.126 | 3.853 |
| r4 | 9.5052 | 15.2861 | 11.071 | 3.860 |

Median slope **11.056 tok/s**, spread 0.835%.

The slope implies a fixed cost of about **3.72 s per `generate()` call**. The
independently measured 1-token call reads **3.85 s**, which is that same fixed
cost plus one token. Two separate readings of the overhead agree to about 4%, so
the linear model the slope assumes is at least consistent with the data. It rests
on two token counts and is not a substitute for measuring decode directly.

That overhead is what makes vLLM's 64-token whole-completion figure 6.734 rather
than 11.056. **This page does not attribute it.** Whether it is prefill of a
5-token prompt, scheduling, detokenization or synchronization was not measured.

### Per-leg readings

**llama.cpp**, `tok/s` and the clock window over each leg:

| leg | tok/s | 3 reps | sclk MHz mean | busy % mean |
|---|---|---|---|---|
| r1 | 12.2296 | 12.2996 / 12.1756 / 12.2136 | 1924.9 | 45.9 |
| r2 | 12.2133 | 12.2632 / 12.1841 / 12.1927 | 2062.8 | 66.4 |
| r3 | 12.2221 | 12.2796 / 12.2139 / 12.1729 | 2079.2 | 65.7 |
| r4 | 12.2163 | 12.2068 / 12.2336 / 12.2084 | 2097.4 | 67.3 |

Median 12.2192, spread 0.133% of the median.

**vLLM**, warm runs only:

| leg | whole-completion tok/s | sclk MHz mean | busy % mean | load s |
|---|---|---|---|---|
| r1 | 6.7302 | 1736.2 | 51.3 | 527.4 |
| r2 | 6.7431 | 1855.3 | 56.8 | 491.1 |
| r3 | 6.7355 | 1905.0 | 58.8 | 476.5 |
| r4 | 6.7332 | 2030.4 | 64.8 | 429.0 |

Median 6.7344, spread 0.192%. Across all 12 warm repetitions: median 6.7362,
range 6.7235 to 6.7656.

The `sclk` means climb across rounds on both arms, from 1925 to 2097 MHz on
llama.cpp and 1736 to 2030 MHz on vLLM, while the throughput of both moves by
less than 0.2%. The board warms over the session and neither figure follows it.

## The vllm.cpp arm: the published fault was a stale binary, and the re-measurement

Lease 1 reported our arm as `FAULTED, 0 of 4, rc 139`, all four legs dying the
same way:

```text
[vt op-provider] op=... device=5 selected=vt-native   (17 to 21 ops)
HW Exception by GPU node-1 (Agent handle: 0x...) reason :GPU Hang
```

**That row was never an engine verdict.** The binary measured was built from
`11fed3ba56b8f823c07032416982a44a8c0967b5`, an **ancestor of `6b97a6800`**, the
commit that fixed exactly this hang by refusing `hipMallocManaged` on a part
that reports `PageableMemoryAccess = 0`. That fix reached `main` as `27da7787e`
at 2026-09-02 15:11 UTC and lease 1 started 2026-09-04 22:29 UTC, two days and
seven hours later, without it. `survey.sh` asserted the llama.cpp source
manifest and refused to run when it moved, but for our own arm it checked that
a file was executable and then **printed** the revision. A recorded value no
gate reads is a comment. That gap is
[#2933](https://github.com/mudler/vllm.cpp/issues/2933); the fault itself and
its fix are [#2511](https://github.com/mudler/vllm.cpp/issues/2511).

### The second lease asserts the head five ways

Lease 2 runs the harness that [#2953](https://github.com/mudler/vllm.cpp/pull/2953)
landed. It asserts the tree under test fail-closed, each link with its own exit
code, before the board is touched:

| link | assertion | reads |
|---|---|---|
| bundle | sha256 of the staged `repo.bundle` | `044dec44…6ca2b8` |
| revision | `rev-parse HEAD` | `c796fea41f74fe90b8cf78190eeb2a5b4c977449` |
| tree | `rev-parse HEAD^{tree}` | `018178f3bf1fca3983d9bc0cfd42f5ea4bf130b1` |
| clean | `git status --porcelain` is empty | 0 modified files |
| built bytes | binaries post-date a build directory deleted before configure | `mtime 1788640323` vs `build_start 1788640061` |
| built bytes | `libvllm.so` carries `cannot take a recoverable page fault` | present |
| built bytes | `vllm-cli` sha256 is **not** `a703b83d…d646f` | `043166a8…6d3d24` |

`job.log.txt` records `REVISION_ASSERTION=OK  TREE_ASSERTION=OK  CLEAN=OK
SOURCE_CARRIES_FIX=OK` and then `BUILT_BYTES_ASSERTION=OK`, with
`BUILD_TARGET_RC=0` printed by a sentinel immediately after the build. The
first four links were also exercised against the real staged bundle before the
lease, in
`docs/bench-evidence/qwen38-27b-q4km-gfx1151-ourarm-head-20260905/assertion-exercise.txt`,
which drives the #2933 scenario backwards: declare the pre-fix revision and the
job exits 24 rather than measuring it. That file is named rather than linked,
because it reached `main` with #2953 and this page's own branch does not carry
it yet.

The two content links are independently checkable from this repository.
`c796fea41` carries the literal `cannot take a recoverable page fault` at
`include/vt/rocm/rocm_arch.h:158` and `11fed3ba5` does not, and
`043166a8…6d3d24` is not `a703b83d…d646f`.

`HSA_OVERRIDE_GFX_VERSION` read `UNSET` and no `HSA_`, `ROCR_`, `PYTORCH_`,
`HIP_`, `GGML_` or `VT_` variable was inherited.

### What it measured

Four rounds, each running a 64-token and a 128-token completion of the same
prompt, with the order of the two token counts rotating by round. Four
repetitions per leg, of which run 1 is cold and is discarded. A clock sampler
ran at 4 Hz beside every leg.

| arm | legs | fault rate | quantity | figure | spread |
|---|---|---|---|---|---|
| n=64 | 4 of 4 | 0 of 4 | whole completion | **5.305 tok/s** | 0.829% |
| n=128 | 4 of 4 | 0 of 4 | whole completion | **5.3535 tok/s** | 0.243% |
| slope | 4 of 4 pairs | — | decode, **derived** | **5.397 tok/s** | 0.564% |

`reference_tier_lines` read **0 on every one of the eight legs**, so no host
fallback ran and every op resolved on the device.

The completion timings below retain warm repetitions only. The clock and busy
means use the whole leg, including model loading, the cold completion, warm
completions, and teardown. Those columns do not share the timing window.

| leg | median secs | whole-completion tok/s | sclk MHz mean | busy % mean |
|---|---|---|---|---|
| n64-r1 | 11.979 | 5.343 | 2050.6 | 63.1 |
| n64-r2 | 12.062 | 5.306 | 2044.3 | 63.2 |
| n64-r3 | 12.078 | 5.299 | 2045.2 | 63.3 |
| n64-r4 | 12.067 | 5.304 | 2045.7 | 63.3 |
| n128-r1 | 23.885 | 5.359 | 2337.6 | 76.0 |
| n128-r2 | 23.905 | 5.355 | 2312.8 | 75.1 |
| n128-r3 | 23.917 | 5.352 | 2350.7 | 76.8 |
| n128-r4 | 23.942 | 5.346 | 2360.2 | 77.4 |

Pooled over all 3,216 whole-leg clock samples the board read **2228 MHz mean at
71.3 percent busy**. The 64-token whole-leg means sit near 2045 MHz and 63
percent and the 128-token means near 2340 MHz and 76 percent. These historical
values include time outside generation and do not establish different clock
states during warm generation.

Joining the same samples to the recorded `generate_start_unix` and
`generate_end_unix` timestamps gives the warm windows below. Retain each
sample whose timestamp falls inside the inclusive bounds of generation 2, 3,
or 4. These windows include prefill and generation, not decode alone.

| leg | warm samples | warm sclk MHz mean | warm busy % mean |
|---|---|---|---|
| n64-r1 | 143 | 2888.6 | 100.0 |
| n64-r2 | 144 | 2874.9 | 100.0 |
| n64-r3 | 144 | 2869.8 | 100.0 |
| n64-r4 | 143 | 2871.3 | 100.0 |
| n128-r1 | 285 | 2877.4 | 100.0 |
| n128-r2 | 281 | 2871.5 | 100.0 |
| n128-r3 | 279 | 2870.0 | 100.0 |
| n128-r4 | 284 | 2865.5 | 100.0 |

The 1,703 warm samples pool to **2872.8 MHz mean and 100.0 percent busy**.
Busy percentage reports sampled GPU activity. It is not occupancy and gives
no quantitative bound on host idle time between samples. `TOKEN_GATE=FAIL`
remains carried from the survey. This clock correction changes no historical
throughput or correctness result.

Reproduce both windows and each generation from the committed captures:

```sh
python3 docs/bench-evidence/qwen38-27b-q4km-gfx1151-ourarm-head-20260905/clock_windows.py
```

The completions themselves are byte-identical across all four legs at each
token count, and the 128-token completion continues the 64-token one exactly,
so the slope below is taken along one trajectory rather than across two.

### The decode figure is DERIVED, and is labelled so wherever it appears

`vllm-cli` reports `tok_s` as `completion_tokens` over the wall time of the
whole `vllm_complete()` call. That is **whole completion**, prompt included. It
is not what `llama-bench -p 0` reports. The slope between two token counts on
the same prompt in the same lease removes whatever fixed cost each call pays:

| pair | t(64) s | t(128) s | slope tok/s | implied fixed cost s |
|---|---|---|---|---|
| r1 | 11.979 | 23.885 | 5.375 | 0.073 |
| r2 | 12.062 | 23.905 | 5.404 | 0.219 |
| r3 | 12.078 | 23.917 | 5.406 | 0.239 |
| r4 | 12.067 | 23.942 | 5.389 | 0.192 |

Median slope **5.397 tok/s**, spread 0.564 percent, 4 of 4 pairs usable. The
implied fixed cost is about **0.205 s per call**, which is why the
whole-completion figure at 64 tokens (5.305) sits below the same run's derived
decode figure (5.397) and why the 128-token figure (5.3535) sits between them.
This is a derivation from two token counts. It is not a direct decode reading,
and this page does not present it as one.

### The two ratios, and the one this page refuses to compute

| basis | comparison | value |
|---|---|---|
| whole completion vs whole completion | vLLM 6.734 / vllm.cpp 5.305 | **1.269x**, vLLM faster |
| decode vs decode, ours **derived** | llama.cpp 12.219 / vllm.cpp 5.397 | **2.264x**, llama.cpp faster |
| pure decode vs whole completion | llama.cpp 12.219 / vllm.cpp 5.305 | **not computed** |

**The third row is refused, not missing.** Dividing `llama-bench -p 0` by a
whole `vllm_complete()` call compares two different quantities. This page
already records what happens when that is done anyway, in the 1.814x the survey
fold emitted for llama.cpp against vLLM. The fold for lease 2 refuses the same
division by name.

**llama.cpp's 12.219 and vLLM's 6.734 are constants carried from lease 1.** No
leg of lease 2 re-measured either of them. Lease 2 measured one engine.

### What the gap is, and what is not established about it

The 2.264x decode gap has a named candidate.
[#2109](https://github.com/mudler/vllm.cpp/issues/2109) attributes 61.5 percent
of GPU time to our ROCm keep-quant GEMM `KQuantGemmK`, against llama.cpp's
`mul_mat_q`, and finds it has no tensor-core arm. **That measurement was taken
on `gfx1200` (RX 9060 XT), not on this board**, by same-tool `rocprofv3`
attribution on a different model.

That issue's title prescribes MFMA and its own comment thread corrects it:
MFMA is CDNA-only, and neither `gfx1200` nor `gfx1151` is CDNA. llama.cpp's
`ggml/src/ggml-cuda/common.cuh:76` states it outright. `gfx1151` is RDNA 3.5,
so the arm to port is llama.cpp's **RDNA3 WMMA** branch at `mma.cuh:729`, not
the RDNA4 tile and not MFMA. No such arm exists in our tree on any RDNA target.

The measured head does carry the ladder's first rung. `f3bb8243b` replaced the
scalar `Dp4a` expansion with `__ockl_sdot4`, for which the contributor measured
1.26 to 1.35x **on `gfx1100`**. The call is unconditional, so it compiled into
this binary, but whether the compiler emits the hardware `v_dot4_i32_i8` on
`gfx1151` and what it is worth there is **not established here**. Nothing on
this page measures either rung on this board.

## The end-to-end control arm was lost

`llama-cli` was in the design so that llama.cpp would have a row on the same
whole-completion definition as vLLM, rather than the survey comparing pure
decode against a whole request. It ignored `-no-cnv`, opened its interactive
chat UI, answered the prompt, and then printed a `> ` prompt against
`/dev/null` until the 25-minute timeout, producing about 24 GB per leg. Mean
`sclk` was 606 MHz at 0.2 percent busy, so the board was idle throughout. That is
[#2935](https://github.com/mudler/vllm.cpp/issues/2935).

**It does not affect the llama.cpp figure**, which comes from `llama-bench`, a
different binary that completed 4 of 4. What it costs is the matched-definition
row, which is why the 1.105x above had to be derived from vLLM's slope instead of
read off a llama.cpp leg.

Each leg did print its own rate before it hung: generation 12.2, 12.2, 12.2 and
12.1 t/s. That is llama.cpp's self-reported figure at one decimal place, and it
agrees with `llama-bench`. It corroborates and is not counted as a leg.

## Agreement with the landed denominator

The landed llama.cpp denominator is **12.233 tok/s**, median of 6 legs, spread
0.303%, range 12.218 to 12.255, taken in a different lease on 2026-09-02.

This run reads **12.219**, median of 4, spread 0.133%, range 12.213 to 12.230.

The two differ by 0.11% and their ranges overlap between 12.218 and 12.230. They
agree. The landed figure stands; nothing here supersedes it.

## Limitations

- **The vllm.cpp figure comes from a different lease than the other two.** Lease
  2 ran our arm alone, a day after lease 1, on the same board, the same boot and
  the same artifact bytes, but not interleaved with the other engines. That
  removes the interleaving control lease 1 had.
- **The vllm.cpp decode figure is derived, not measured.** It is the slope
  between a 64-token and a 128-token completion of the same prompt and it
  assumes the cost is linear in tokens. Only the whole-completion figures,
  5.305 and 5.3535 tok/s, are direct readings.
- **llama.cpp's 12.219 and vLLM's 6.734 are constants carried from lease 1.**
  Lease 2 re-measured neither of them.
- The gap's named candidate, #2109, was measured on `gfx1200` and on a different
  model. Nothing on this page attributes this board's gap to any kernel.
- **Neither lease recorded how many layers each side ran, so the 2.264x carries
  an unmeasured configuration term.**
  [#2497](https://github.com/mudler/vllm.cpp/issues/2497) names this as an
  acceptance condition and it is not met here. For the Qwen3.8-27B Q4_K family
  that issue records llama.cpp at `b10451` loading 851 of 866 tensors and
  ignoring all 15 of `blk.64`, four of them the `nextn.*` MTP head, against our
  loader reading `qwen35.block_count = 65` with
  `qwen35.nextn_predict_layers = 1`. An arm that runs block 64 does strictly
  more work per token than this denominator. Whether that difference is present
  in these two leases is not recorded in either of their logs.
- The four correctness counts are transcribed from earlier evidence and were not
  re-measured by either lease.
- The decode-only vLLM figure is derived from two token counts and assumes the
  cost is linear in tokens. It is not a direct decode measurement.
- The matched whole-completion comparison has no llama.cpp side at all, because
  that arm timed out.
- vLLM ran on the host out of its own venv while the other arms ran in a
  container. Both open `/dev/kfd` directly. The asymmetry is recorded, not
  corrected.
- llama.cpp's custody chain has one open link, carried forward: the source is
  pinned by content manifest and the binaries are pinned as bytes, but no
  compiler ran in this lease.
- This historical survey used an ad-hoc AMD clock sampler. The subsequent
  [#3015 worker](../../tools/bench/strix_kernel_trace/worker.py) samples AMD
  clock state and folds it into recorded warm generation windows.

## Evidence

`docs/bench-evidence/qwen38-27b-q4km-gfx1151-survey-20260904/` holds the job
log, the three harness scripts, the clock sampler, the per-leg captures and the
fold's `RESULT`. JSON captures carry a `.json.txt` suffix so that they classify
as evidence rather than as public documents.

`docs/bench-evidence/qwen38-27b-q4km-gfx1151-ourarm-head-20260905/` holds lease
2: the spec's harness and its assertion exercise, landed by
[#2953](https://github.com/mudler/vllm.cpp/pull/2953), and this run's
`job.log.txt`, `build.log.txt`, `RESULT.json.txt`, the per-leg captures and
return codes, the gzipped clock samples, and `rederive.py` with its
`rederivation.txt`. The `.log` captures carry a `.txt` suffix because
`.gitignore` excludes `*.log`.

`rederive.py` re-derives the original lease-2 figures from the raw per-leg
`.err` and `.jsonl` records alone and then compares the result with the fold's
`RESULT.json`: **16 checks, 0 mismatches**. It runs against the share when the
share is reachable and against the committed copies in its own directory when
it is not, and `rederivation.txt` records that the two runs agree line for
line. The adjacent `clock_windows.py` reproduces the warm clock correction
from the committed captures, without selecting an external share. It also
prints the original whole-leg clock means and each generation's window.
Both commands preserve the original `RESULT.json` comparison contract.

Raw per-leg artefacts, including the 24 GB `llama-cli` captures which are not
committed, remain at `/mnt/nas_share/rc/strix-survey-2497/out/survey-20260904/`
for lease 1 and at `/mnt/nas_share/rc/strix-arm-2933/out/` for lease 2.

Issues: [#2921](https://github.com/mudler/vllm.cpp/issues/2921) the survey,
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) the missing quant-matched
number, [#2933](https://github.com/mudler/vllm.cpp/issues/2933) the unasserted
revision, [#2944](https://github.com/mudler/vllm.cpp/issues/2944) the
re-measurement on an asserted head,
[#2935](https://github.com/mudler/vllm.cpp/issues/2935) the control arm, and
[#2109](https://github.com/mudler/vllm.cpp/issues/2109) the keep-quant GEMM that
has no tensor-core arm on any RDNA target.
