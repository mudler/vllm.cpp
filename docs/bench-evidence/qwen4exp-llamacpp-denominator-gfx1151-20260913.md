# The llama.cpp denominator for Qwen3.8-Flash-Next UD-IQ1_S on `gfx1151`, 2026-09-13

**`25.877 tok/s`**, median of 12 legs, leg spread `1.734%` of the median.

One engine, measured alone. Row `MODEL-MM-QWEN4-EXP`, issue
[#2060](https://github.com/mudler/vllm.cpp/issues/2060) (which owns the missing
llama.cpp denominator) and the new local issue
[`ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG`](../../.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG.md)
(which owns the missing precondition for any ratio). Oracle
[`llama-cpp-qwen4exp`](../../.agents/oracles/llama-cpp-qwen4exp.md). Artifacts:
[`qwen4exp-llamacpp-denominator-gfx1151-20260913/`](qwen4exp-llamacpp-denominator-gfx1151-20260913/).

## What this is, and what it is not

**It is a denominator.** llama.cpp at the `llama-cpp-qwen4exp` pin, built with
HIP for `gfx1151`, decoding one byte-verified `Qwen3.8-Flash-Next-UD-IQ1_S`
GGUF on `strix:gpu0`. One engine's own speed on one board is a single-engine
fact, and it needs no second engine to be true.

**It is not a comparison, and no ratio appears anywhere in this document.**
`AGENTS.md` §Gates admits a performance result from an arm only after that arm's
declared token-exact gate passes. The `qwen4_exp` ROCm arm has **no such gate —
not a failing one, an absent one**. **THE REASON THIS FILE FIRST GAVE FOR THAT
IS FALSE AND IS REPLACED RATHER THAN AMENDED** (corrected 2026-09-13, after the
measurement and without touching it). It read: "it cannot simply be written,
because `Qwen4ExpForConditionalGeneration` / `qwen4_exp` is an architecture no
vLLM revision implements ([#1978](https://github.com/mudler/vllm.cpp/issues/1978)),
so the primary oracle cannot define what token-exact means here." vLLM
implements this architecture in full at the ACTIVE parity pin `e126687a9a`
(`[Model] Support Qwen3.8-Flash-Next`, vllm#53896), with
`Qwen4ExpForConditionalGeneration` registered at
`vllm/model_executor/models/registry.py:580`, so the primary oracle CAN define
token-exact here. [#1978](https://github.com/mudler/vllm.cpp/issues/1978) is not
wrong, it is EXPIRED: it was read live against vLLM `origin/main` = `6a5e8f5979`
on 2026-08-26, five days before that support landed. **What is missing is a
primary-oracle RUN, and the blockers are specific**: `cooperative_topk` in the
QSA indexer refuses to launch on this fleet at that revision
([#2626](https://github.com/mudler/vllm.cpp/issues/2626), cause unestablished);
every published safetensors arm exceeds the largest fleet box and vLLM cannot
open the GGUF this file measured; and the decode GEMM plan is `sm_103`-gated.
[`ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG`](../../.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG.md)
owns that gap. **Nothing above changes this document's refusal to divide**, which
rested on the ABSENT gate and never on the absent oracle. This run
therefore produced **no vllm.cpp figure at all**:
`vllmcpp_binaries_built=0`, `vllmcpp_legs_run=0`, `ratios_computed=0`, recorded
in `RESULT.json` by the job itself.

This follows the shape and the refusal of
[`rocm-strix-llamacpp-denominator-20260902.md`](rocm-strix-llamacpp-denominator-20260902.md),
which took the same measurement for a different model on the same board and
declined to divide for the same reason.
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) already carries one
retraction for taking a cross-engine number ahead of that gate.

**Neither side is token-gated here, and that is two separate facts.** Ours has
no gate. llama.cpp's own decode at this pin is not gated against anything
either, and [`llama-cpp.md`](../../.agents/oracles/llama-cpp.md) records that
this oracle's greedy decode is not deterministic across its own kernel paths.

**The two engines do not serve the same surface.** `/props` at this pin reports
`"modalities":{"vision":false,"video":false,"audio":false}` — this denominator
is **TEXT ONLY**, while `MODEL-MM-QWEN4-EXP` is a multimodal port. An arm that
runs a vision tower does strictly more work per request.

## The figure

| | tok/s |
|---|---|
| **Median of 12 legs** | **25.877** |
| Mean of 12 legs | 25.882 |
| Min / max leg | 25.687 / 26.135 |
| Leg spread | 1.734% of the median |
| Standard deviation of the legs | 0.142 |
| Coefficient of variation | 0.550% |
| Median of the 36 underlying repetitions | 26.096 |
| Min / max repetition | 24.521 / 26.782 |
| Repetition spread | 8.666% of their median |

`N = 12` is the **design**, handed to the fold as the loop bound and read back
by the fold from `leg1.json … leg12.json`. It is never derived by counting log
lines: the per-leg figures are echoed twice into the same log, once by the leg
and once by `RESULT.json`, so a `grep -c` over that log answers the wrong
question, exactly as the 2026-09-02 record measured.

### Per leg, in the order they ran

A leg is one `llama-bench` process: one model load, `llama-bench`'s own warmup
generation, then 3 timed generations of 64 tokens. The per-leg figure is that
leg's own `avg_ts`, straight out of its JSON.

| Leg | rc | wall | `avg_ts` | `stddev_ts` | the 3 repetitions |
|---|---|---|---|---|---|
| 1 | 0 | 40 s | 26.1353 | 0.5543 | 26.1469 / 26.6838 / 25.5754 |
| 2 | 0 | 45 s | 25.7425 | 0.5550 | 26.0000 / 26.1219 / 25.1055 |
| 3 | 0 | 49 s | 25.8831 | 0.5348 | 26.1956 / 26.1881 / 25.2656 |
| 4 | 0 | 45 s | 25.7338 | 0.7274 | 26.1258 / 26.1811 / 24.8945 |
| 5 | 0 | 47 s | 25.9175 | 0.5908 | 26.1201 / 26.3804 / 25.2521 |
| 6 | 0 | 48 s | 25.8716 | 0.9097 | 25.9604 / 26.7337 / 24.9208 |
| 7 | 0 | 49 s | 26.0039 | 0.7952 | 26.0367 / 26.7821 / 25.1928 |
| 8 | 0 | 49 s | 26.0359 | 0.6428 | 26.0912 / 26.6493 / 25.3673 |
| 9 | 0 | 49 s | 25.6868 | 1.0381 | 26.0301 / 26.5097 / 24.5205 |
| 10 | 0 | 46 s | 25.9849 | 0.4814 | 25.9718 / 26.4727 / 25.5102 |
| 11 | 0 | 46 s | 25.8695 | 0.8667 | 26.2438 / 26.4861 / 24.8785 |
| 12 | 0 | 48 s | 25.7140 | 0.8265 | 26.1009 / 26.2760 / 24.7651 |

**12 of 12 legs completed and none was discarded.** No leg hit the `gfx1151`
illegal-memory-access signature
([`ISSUE-LOCAL-01M2BY2M2ATNVR3XQKV2DB1BJD`](../../.agents/issues/BACKEND-ROCM/ISSUE-LOCAL-01M2BY2M2ATNVR3XQKV2DB1BJD.md), about two
runs in five on that board); the job carries an explicit check for it and it
never fired, across **31** `llama-bench` processes in four leases: 1 pilot + 6 legs, 12 legs + 4 at `n_gen=32`, 1 loader capture, and 7 placement probes.

The order column carries no monotonic drift: the fastest leg ran first, the
slowest ran ninth, and the second-fastest ran eighth.

### This reproduces across leases

A predecessor population of 6 legs ran in a **different `rc` lease** 9 minutes
earlier, same binary, same artifact, same board, same boot:

| | 6-leg predecessor | 12-leg primary |
|---|---|---|
| median | 25.6864 | **25.8774** |
| mean | 25.5324 | 25.8816 |
| min / max leg | 24.4666 / 26.0411 | 25.6868 / 26.1353 |
| leg spread | **6.130%** | 1.734% |
| repetitions | 18 | 36 |

The two medians are **0.744% apart**, which is the second thing a denominator
has to show after agreeing with itself. Raw legs:
the `run1-*` files in
[`qwen4exp-llamacpp-denominator-gfx1151-20260913/`](qwen4exp-llamacpp-denominator-gfx1151-20260913/).

**The predecessor's 6.130% spread is reported, not explained away.** One leg,
its third, reads 24.4666 and is the population minimum by a clear margin: ranked
slowest to fastest the six legs read 3, 5, 2, 6, 4, 1 and the successive gaps
are 1.1124, 0.1021, 0.0106, 0.0432, 0.3063 tok/s — the largest gap in that
population sits *below* leg 3 and separates it from the other five. Drop it and
the remaining five span 1.798%, which is the primary run's band. The primary run
began 9 minutes later on a quieter box (`loadavg` 1-minute 1.46 versus 0.39 with
a 15-minute of 6.56, the tail of this row's own HIP build job that had finished
two minutes before the predecessor started). **That is a hypothesis and not a
measured cause, so the predecessor is quoted whole, including leg 3, and the
headline is the 12-leg population that was taken on the quiet box.**

### A secondary set at `n_gen = 32`

The row's other, separately-reported arm generates 32 tokens, so 4 legs were
taken at that length. They are recorded so that a future gated comparison does
not have to re-derive them. **They are not divided by anything here.**

| | tok/s |
|---|---|
| Median of 4 legs | 25.810 |
| Mean / min / max | 25.839 / 25.653 / 26.084 |
| Leg spread | 1.668% |
| Repetitions | 12, median 25.845 |

## Pinning the EXECUTED PATH, not only the revision

| Term | Value |
|---|---|
| Revision | `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, `ggml-org/llama.cpp` PR #27742 |
| `git rev-parse HEAD` in the build | asserted **equal** to the pin, or the job exits 94 |
| `git status --porcelain` | **0 bytes** |
| `build_commit` / `build_number`, from each leg's own JSON | `035e227` / `1` |
| `/props` `build_info` | `b1-035e227` |
| `system_fingerprint` on the completion | `b1-035e227` |
| Source content manifest (pre-build, `LC_ALL=C`) | `b76b8fcae18c5c7b25e08b756817a518adc1c36400136255976fa43ca8f9fa95` |
| `llama-bench` | `03bc5011bdafaa82eaeca4f32ce0ecf35f145b2ab20c79d8f0e69502596a1597` |
| `llama-cli` | `8617aa73d9694e5d8f2fc782763f6be4ac7ed9a2bce10bb28bd919b0326650e9` |
| `llama-server` | `222e5c795a87343156979a450ea0c181dc1c65c4dccde745c153fb2f767384c2` |
| `libllama.so.0.3.0` | `a7ac5b5f4038c26edeff2e5ff34716d20609981b12dbebda178eece5735fb30c` |
| `libggml-hip.so.0.22.0` | `6e1149e6e0f5700049c7b066380376e23fe99051dcc0ba8086b05e9239e167e4` |
| `libggml-cpu.so.0.22.0` | `8dc5f52d4e073e04936f02d51905b6d281fb14a44c73cb005d611f926c79fc06` |
| `libggml.so.0.22.0` | `e158e4db3745d249d69a1a394af20f532b54d3deedd4fbdeee756c42a4f18138` |
| `libggml-base.so.0.22.0` | `94c5394de87fa2bbc2b345e02f8544e88d29754feefc3af600de79999059c936` |
| Backend, per leg's own JSON | `ROCm` |
| `n_gpu_layers` / `n_cpu_moe` / `split_mode` | `99` / `0` / `layer` |
| `n_gen` / `n_prompt` / `n_depth` | `64` / `0` / `0` |
| `n_threads` | `16`, of `nproc` 32 |
| `type_k` / `type_v` / `flash_attn` | `f16` / `f16` / `-1` (auto) |
| `model_type` | `qwen4exp A3B IQ1_S - 1.5625 bpw` |
| `model_size` / `model_n_params` | 72,535,436,800 / 176,943,899,520 |
| `cpu_info` / `gpu_info` | `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S` / `AMD Radeon Graphics` |
| Enumerated device | `ROCm0: AMD Radeon Graphics (98304 MiB, 98148 MiB free)`, `gfx1151 (0x1151)`, `Wave Size: 32`, `VMM: no` |

**Unlike the 2026-09-02 record, the content-to-binary link is CLOSED here.**
That run staged a tarball, found the binaries already present and recorded
`llamacpp_build=ALREADY PRESENT`, so no compiler ran in its lease. This tree was
fetched by SHA and compiled in a lease of this campaign, and every leg asserted
the resulting `llama-bench` sha256 before running. The one link that stays open
is the ordinary one: the `sha256` of a binary does not by itself prove which
source produced it, which is why the build job's log, the configure log and the
manifest are committed beside the numbers.

**One harness defect, stated because its output is in the tree.** The build job
computed its post-build source manifest over a path that contains the build
directory, so `llama_src_manifest_after_build` reads
`ea845de3d1f3c4c0842252ae1b068f352b006e35e9563bd7354633f3ef40ffef` and differs
from the pre-build value. **That is the harness measuring its own object files,
not a dirtied source tree.** The pre-build manifest is the clean one, and
`git status --porcelain` was read at 0 bytes at the pin.

## It is a HIP build, proven four ways

1. **CMake was asked for it.** `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151
   -DGGML_HIP_ROCWMMA_FATTN=OFF
   -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
   -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-7.2.4`, Release, `-j 4`.
2. **`ldd` resolves the ROCm runtime.** `llama-bench`, `llama-cli` and
   `llama-server` each resolve `libamdhip64.so.7`, `libhsa-runtime64.so.1`,
   `libhipblas.so.3`, `librocblas.so.5` and `libhipblaslt.so.1`, all from
   `/opt/rocm/lib`, with no `not found`. The job refuses on an unresolved
   library rather than measuring a 127.
3. **The device code is for this chip.** `libggml-hip.so.0.22.0` carries **270**
   `gfx1151` strings.
4. **The architecture is linked, not merely in source.** `libllama.so.0.3.0`
   carries **76** `qwen4exp` strings, and the loader prints
   `general.architecture str = qwen4exp`, `print_info: arch = qwen4exp`.

Toolchain: ROCm 7.2.4, HIP `7.2.53211-97f5574fe2`, AMD clang 22.0.0git
(`roc-7.2.4`), on the `rc` worker image (`hipcc`, `rocminfo`, `rocblas-dev` and
`hipblas-dev` already present — no install was needed, unlike the CUDA recipe).

**The HIP build needed no patch at the pin.** `scripts/qwen4exp-llamacpp-build-cuda.sh`
is the CUDA recipe this one mirrors; the HIP equivalent as run is committed as
[`build-hip-job.sh`](qwen4exp-llamacpp-denominator-gfx1151-20260913/build-hip-job.sh).

## It decodes, and it decodes coherently

`llama-server` at this pin, on this artifact, answered `/health` after 40 s and
returned to `/v1/completions` for the prompt `The capital of France is`, greedy,
`seed 1`:

```
" Paris. Given a sequence of letters, generate the capital of the country.\n"
```

`finish_reason=length`, 16 completion tokens, 5 prompt tokens,
`"predicted_per_second": 24.79`.

## WHERE THE TENSORS ACTUALLY LANDED — and why the default split is the fast one

**A `-ngl 99` that silently fell back to the CPU would still print a number.**
Measured from the loader's own `-v` output on the same binary that produced every
leg:

```
load_tensors: offloading output layer to GPU
load_tensors: offloading 47 repeating layers to GPU
load_tensors: offloaded 49/49 layers to GPU
load_tensors:          CPU model buffer size = 27465.95 MiB
load_tensors:        ROCm0 model buffer size = 41368.28 MiB
load_tensors:    ROCm_Host model buffer size =   341.02 MiB
```

Every one of the 48 layers reads `assigned to device ROCm0`, and llama.cpp calls
this `49/49 layers offloaded` — **and 27,465.95 MiB of weights are nonetheless
in a CPU buffer.** 41,368.28 + 27,465.95 + 341.02 = 69,175.25 MiB, the whole
67.55 GiB file. So the denominator is a **hybrid placement**, and a reader who
takes `-ngl 99` to mean "all weights on the GPU" would misread it.

**That split is not a fallback, it is llama.cpp's fast configuration here, and
the alternative was measured rather than assumed.** Five placements, one leg of
3 repetitions each, same binary, same artifact, same lease:

| Probe | `CPU` model buf | `ROCm0` model buf | `avg_ts` |
|---|---|---|---|
| A `-ngl 99` (the denominator's own configuration) | 27,465.95 MiB | 41,368.28 MiB | 26.0099 |
| A2 (repeat) | identical | identical | 25.7447 |
| A3 (repeat) | identical | identical | 25.6222 |
| B `-ot exps=ROCm0` | 27,465.95 MiB | 41,368.28 MiB | 26.1310 |
| **C `-ot .*=ROCm0`** | **absent** | **69,175.25 MiB** | **1.9534** |
| D `--no-host 1` | 27,806.97 MiB | 41,368.28 MiB | 25.7767 |
| E `--mmap 0` | 27,465.95 MiB | 41,368.28 MiB | 25.7735 |

**C is the one that moved the split, and it is 13x slower.** Forcing every
tensor into `ROCm0` produces the full-GPU residency a reader might expect from
`-ngl 99` and collapses decode to 1.95 tok/s. **B is the control that proves the
table is reading a real term and not noise**: its `-ot` pattern did not match, it
reproduced A's buffer split byte-for-byte, and it landed inside A's own band. So
the default is the production configuration on this board, and no leg here
crippled the oracle to flatter anything — `AGENTS.md` §Gates requires exactly
that, and this is the measurement that discharges it.

A3 at 25.6222 sits 0.251% below the 12-leg minimum, which places these
single-leg probes at the edge of, not inside, the primary population's band.
They are quoted only to rank the placements against each other, never as
additional legs.

### KV and recurrent state, which [#2261](https://github.com/mudler/vllm.cpp/issues/2261) owes a measurement of

`llama-server` prints no KV sizing line, which is what that issue records. The
`-v` path on `llama-bench` **does**, and at `n_ctx = 256`, `n_seq_max = 1`:

```
llama_kv_cache: size =   6.00 MiB ( 256 cells, 12 layers, 1/1 seqs), K (f16): 3.00 MiB, V (f16): 3.00 MiB
llama_kv_cache: size =   2.25 MiB ( 256 cells, 12 layers, 1/1 seqs), K (f16): 0.75 MiB, V (f16): 1.50 MiB
llama_memory_recurrent: size = 124.88 MiB (1 cells, 48 layers, 1 seqs), R (f32): 16.88 MiB, S (f32): 108.00 MiB
```

Only 24 of 48 layers carry a KV cache at all — the rest read
`llama_kv_cache: layer N: filtered` and appear instead in the recurrent state,
which is the hybrid linear-attention shape of this architecture. **This is a
partial answer to #2261 and is not offered as a closing one**: the issue owes a
per-token cost measured under `-np 32 -c 49152`, and this is one slot at
`n_ctx = 256`. It does establish that the sizing lines exist on a path that is
not `llama-server`'s stdout.

Compute buffers: `ROCm0` 7.03 MiB, `ROCm_Host` 0.08 MiB, `graph nodes = 7269`,
`graph splits = 3`. Load time on the local copy: `34,468 ms` for the first
context.

## The artifact, and the filesystem it was read from

| Term | Value |
|---|---|
| Opened file | `Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf`, shard 1 of 3 |
| sha256 of shard 1 | `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`, asserted by every job before it measured |
| Shard sizes | 10,946,624 + 49,990,818,368 + 22,544,696,352 bytes |
| `print_info: file size` | 67.55 GiB (3.28 BPW) |
| Path | `/tmp/ckpt-iq1s`, the worker's **local overlay** (1.9T, 21% used) |
| Metadata | `general.architecture = qwen4exp`, 67 key-value pairs, 1224 tensors, 2 additional GGUFs |
| Architecture terms | `block_count 48`, `embedding_length 2560`, `head_count 24`, `head_count_kv 2`, `expert_count 512`, `expert_used_count 10`, `context_length 262144` |

**The CIFS confound is excluded by construction.** The share strips the execute
bit and holds no symlink, and on this board 657 of 696 s of a first load was
page-cache read wait on it. Every timed leg read the local copy. The build was
made in `/tmp` and archived to `/workspace` as a **tar**, because
`cp`-ing a build tree to the share loses the mode bits and a `find -type f`
copy-out loses the SONAME symlinks — both produce `rc=127` with
`libllama-common.so.0` not found.

## Environment and contention

| Term | Value |
|---|---|
| Device | `strix:gpu0`, AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, `gfx1151`, RDNA 3.5 |
| Reached by | `rc run -d strix:gpu0`, exclusively leased, never `ssh` |
| Jobs | build `--max-runtime 115m`; predecessor `235m`; primary `120m`; placement `40m`; placements `60m` |
| Worker | `rc-worker-lcjhd`, x86_64, `nproc` 32, 30 GiB host RAM |
| Boot id | `2ed44130-3e9a-487f-876c-5cbb57f38aab`, **the same across every leg of both runs** |
| `loadavg` 1/5/15 before the primary run | 1.46 / 2.33 / 4.62 |
| `loadavg` 1/5/15 after the primary run | 3.28 / 3.17 / 3.86 |
| Other fleet devices in use during the run | `dgx:gpu0` (a different row), `thor:gpu0` (a different row) — neither shares this board |

`rc ps` was read before each submission and no other job held `strix:gpu0`. The
lease is the exclusion; the file mutex `$HOME/gpu.lock` is not taken because the
fleet cannot see it, and taking it here would recreate the two-mutex failure
`AGENTS.md` names.

## The SM clock: SAMPLED, never asserted, and it does NOT meet the 5% ceiling

`.agents/benchmarking.md` requires the within-run SM-clock spread to stay at or
below 5%. **This run does not meet that, and the number is printed rather than
omitted.** Pooled over the 363 samples taken while the GPU was at least 50%
busy across the 12 legs: min 1776 MHz, **median 2795 MHz**, max 2869 MHz,
spread **39.106%**. Per leg the spread runs 30.18% to 38.95%.

Three things have to be said about that, in order.

**The clock cannot be pinned inside a lease.** `nvidia-smi -lgc` has no AMD
counterpart reachable here, and the lease's `CapBnd` is the default OCI set with
no `CAP_SYS_ADMIN`
([`specs/lease-gpu-capability.md`](../../.agents/specs/lease-gpu-capability.md),
[#1354](https://github.com/mudler/vllm.cpp/issues/1354)). Fleet devices are
reachable by lease only, so on this board the clock can be **sampled and not
pinned**, and a run can be refused on spread with no lever to fix it.

**The window is not decode-only, and that inflates the spread.** A leg is ~46 s
of wall of which the model load is ~35 s, so most of the window is load. The
figures above already filter to `busy_percent >= 50` to exclude the idle load
phase; the unfiltered `busy > 0` population reaches down to ~620 MHz and reads
~80%. Neither filter is a decode-only window. Building one is owed work.

**What follows from it.** The 5% ceiling exists to make a *cross-arm ratio*
trustworthy, and this document takes no ratio. For a single-engine figure the
consequence is narrower and is stated as the guide states it: a delta smaller
than about 10% between this figure and another is **not established by this
figure alone**. The 0.744% agreement between the two leases is therefore
evidence of reproducibility, not a measurement of a 0.744% difference.

Clock samples: the `run2-clock-leg*.jsonl` files in
[`qwen4exp-llamacpp-denominator-gfx1151-20260913/`](qwen4exp-llamacpp-denominator-gfx1151-20260913/),
taken at 0.25 s by the committed sampler
[`amd_clock_sample.py`](rocm-strix-llamacpp-denominator-20260902/amd_clock_sample.py),
reused verbatim rather than re-rolled.

## Warm-up, treated symmetrically and stated

The row's other arm stages ~72 GiB lazily and its first generation is not a
decode number. **The symmetric treatment on this side is `llama-bench`'s own
warmup generation**, which it runs before the timed repetitions of every leg and
excludes from `samples_ts`. So no figure in this document contains a cold weight
upload, on either count, and each of the 12 legs pays its own model load outside
its own clock.

## What was NOT run, and is UNVERIFIED

- **No vllm.cpp leg.** Deliberate. See "What this is, and what it is not".
- **No token-exactness gate, on either side.** There is no reference
  implementation for `qwen4_exp` to gate against.
- **No prompt-processing figure.** Every leg ran `-p 0`.
- **No multi-slot or long-context figure.** `n_ctx` was `llama-bench`'s own
  sizing (256 for the `-v` probe), one sequence, one slot. The `-np 32 -c 49152`
  measurement #2261 owes is **still owed**.
- **No vision, video or audio leg.** This oracle serves text only at this pin.
- **No decode-only clock window.** See above.
- **No second board.** `gfx1151` only.
- **The `-ngl 99` CPU/GPU buffer split is measured but not EXPLAINED.** Which
  tensors llama.cpp routes to the CPU buffer, and why that is 13x faster than
  full `ROCm0` residency on a unified-memory part, is not established here.

## The committed evidence tree

Every file sits directly in
[`qwen4exp-llamacpp-denominator-gfx1151-20260913/`](qwen4exp-llamacpp-denominator-gfx1151-20260913/),
one level deep, because `scripts/check-pr-size.py` classifies a per-run evidence
directory at exactly one level and refuses a deeper path by name. The four
populations are carried by a filename prefix instead of a subdirectory:
`run2-*` the 12-leg primary, `run1-*` the 6-leg predecessor, `place-*` the
loader and placement capture, `off-*` the five placement probes. The five job
scripts are committed as run, beside their logs.

## Reproducing it

```sh
rc run -d strix:gpu0 --max-runtime 115m -- bash -c "$(cat build-hip-job.sh)"
rc run -d strix:gpu0 --max-runtime 120m -- bash -c "$(cat run2-job.sh)"
```

Both scripts are committed beside this file, as run, together with their logs.
