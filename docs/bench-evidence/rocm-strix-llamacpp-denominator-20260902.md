# The llama.cpp denominator for Qwen3.8-27B Q4_K_M on `gfx1151`, 2026-09-02

**`12.233 tok/s`**, median of 6 legs, spread `0.303%` of the median.

One engine, measured alone. Row `BACKEND-GATE-ROCM-LLAMACPP`, issue
[#2497](https://github.com/mudler/vllm.cpp/issues/2497), spec
[`bench-rocm-llamacpp-denominator.md`](../../.agents/specs/bench-rocm-llamacpp-denominator.md).
Artifacts:
[`rocm-strix-llamacpp-denominator-20260902/`](rocm-strix-llamacpp-denominator-20260902/).

## What this is, and what it is not

**It is a denominator.** llama.cpp `b10451` decoding one byte-verified
`Qwen3.8-27B-Q4_K_M.gguf` on `strix:gpu0`. One engine's own speed on one board
is a single-engine fact, and it needs no second engine to be true.

**It is not a comparison, and no ratio appears anywhere in this document.** Our
ROCm `gfx1151` arm's declared token gate reads `TOKEN_GATE=FAIL` at 3 of 6
prompts
([token gate v2](qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md)), and
`AGENTS.md` §Gates admits a performance result from an arm only after that arm's
declared token-exact gate passes. So this run produced **no vllm.cpp figure at
all**: `vllmcpp_binaries_built=0`, `vllmcpp_legs_run=0`, `ratios_computed=0`, in
the job's own words at step 9. #2497 already carries one retraction for taking a
cross-engine number ahead of that gate, and the way not to repeat it is not to
produce the number.

The paired run is staged and inert at
[`.agents/scripts/rocm-strix-ourarm-staged.sh`](../../.agents/scripts/rocm-strix-ourarm-staged.sh),
which refuses to start until the gate is ratified.

## The figure

| | tok/s |
|---|---|
| **Median of 6 legs** | **12.233** |
| Mean of 6 legs | 12.234 |
| Min / max leg | 12.218 / 12.255 |
| Leg spread | 0.303% of the median |
| Median of the 18 underlying repetitions | 12.223 |
| Min / max repetition | 12.151 / 12.333 |

`N = 6` is the **design**, handed to the fold as `--legs 6`. It is never derived
by counting log lines, and the log says why: `grep -c 'avg_ts' job.log` reads
**13** against six legs. Six of those are the per-leg lines step 7 echoes, six
more are the same figures inside the `RESULT.json` that step 8 echoes back into
the same log, and the thirteenth is the fold's own `population` string. So a
tally over that log answers 13, or 7, or 6, depending only on which term is
grepped, and it is never used as a leg count. (`tee -a` is not the mechanism:
it writes each line once. Re-emission is.)

### Per leg, in the order they ran

A leg is one `llama-bench` process: one model load, `llama-bench`'s own warmup,
then 3 timed generations of 64 tokens. The per-leg figure is that leg's own
`avg_ts`, straight out of its JSON.

| Leg | Order | rc | `avg_ts` | `stddev_ts` | the 3 repetitions |
|---|---|---|---|---|---|
| 1 | 1 | 0 | 12.2176 | 0.0065 | 12.2243 / 12.2171 / 12.2114 |
| 2 | 2 | 0 | 12.2437 | 0.0652 | 12.3190 / 12.2055 / 12.2066 |
| 3 | 3 | 0 | 12.2423 | 0.0558 | 12.3017 / 12.1910 / 12.2344 |
| 4 | 4 | 0 | 12.2547 | 0.0730 | 12.3334 / 12.1893 / 12.2415 |
| 5 | 5 | 0 | 12.2241 | 0.0317 | 12.2457 / 12.2389 / 12.1878 |
| 6 | 6 | 0 | 12.2239 | 0.0739 | 12.2989 / 12.1512 / 12.2216 |

**6 of 6 legs completed, and none was discarded.** The declared discard policy
allowed exactly one named cause — a cold page cache on leg 1 — and it was not
reached for. Leg 1 is the minimum, so the question is whether it is *separated*
from the rest, and it is not. Ranked slowest to fastest the legs read 1, 6, 5,
3, 2, 4, and the successive gaps are **0.0063, 0.0002, 0.0182, 0.0013,
0.0110** tok/s: the largest gap in the population sits in the middle of it, not
below leg 1. A cold-cache leg would stand off the bottom of the ranking, and
this one does not.

The order column also carries no monotonic drift: on that same ranking the
slowest leg ran first, the fastest ran fourth and the second-slowest ran last.

**This reproduces across leases.** The predecessor run of 2026-09-01 took the
same oracle on the same board in a different lease and read 12.239 / 12.242 /
12.249. Those three legs and these six overlap, which is the second thing a
denominator has to show after agreeing with itself.

## Pinning the EXECUTED PATH, not only the revision

[`.agents/oracles/llama-cpp.md`](../../.agents/oracles/llama-cpp.md) records that
this oracle's greedy decode is **not deterministic across its own supported
kernel paths**, so a record that names only `b10451` is under-specified. Every
term below was captured by the job itself.

| Term | Value |
|---|---|
| Revision | `10bf611e533d81f739128304991c5e133c6aebd8`, label `b10451` |
| Source identity | content manifest `56c26d15c2acf11b8621ac26663b4316dc29719d765ba1d95231ffacaddf3cda`, `LC_ALL=C` collation, of the source tree staged in the pod. Read twice, at `job.log:38` and `job.log:53` — **no build ran between those two reads**; see link 2 below |
| `llama-bench` | `e2acbe26f4ef214ca5b3658a1062929242345a1205f9a13d6af841bfa6c3f2c1` |
| `llama-cli` | `d563d9877adb49b4ae2c0a6c23e4019548ffc0c7e2d3a7c4339af3f636646a55` |
| `libllama.so.0.1.0` | `c4b3e80215904fbe833a8ede82cd9357732088f36a2c4b29e27ebdfa8451ae56` |
| `libggml-hip.so.0.20.0` | `778a512b1a764c4288b5e2d86597bb541cc54d6393ce9d95efaece2c1a238050` |
| `libggml-cpu.so.0.20.0` | `7c44f77abf43c71f3e3cb0ac888a370d032b10084f9cb34e0c6f1c9d747aa910` |
| `libggml.so.0.20.0` | `9a04df51f6083d536355b8318615679171f1025a1aef6c6e72c1f0999df4e132` |
| `libggml-base.so.0.20.0` | `1de779ff5e8c3d29f73e0ff37775c36eb1ec69f8fce438419bac07c330a800f6` |
| Backend, per leg's own JSON | `ROCm` |
| `n_gpu_layers` | `99`, echoed back out of `llama-bench`'s JSON |
| `n_gen` / `n_prompt` | `64` / `0` |
| `n_threads` | `16`, of `nproc` 32 |
| `flash_attn` | `-1` (auto) |
| `type_k` / `type_v` | `f16` / `f16` |
| `model_type` | `qwen35 27B Q4_K - Medium`, `model_size` 17,095,778,304 |
| `cpu_info` / `gpu_info` | `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S` / `Radeon 8060S Graphics` |
| `system_info` | `ROCm : NO_VMM = 1`, and on the CPU side `AVX512 = 1 \| AVX512_VBMI = 1 \| AVX512_VNNI = 1 \| AVX512_BF16 = 1 \| LLAMAFILE = 1 \| OPENMP = 1 \| REPACK = 1` |
| Enumerated devices | `ROCm0: Radeon 8060S Graphics (65536 MiB, 59913 MiB free)`, `Wave Size: 32`, `VMM: no` |

**`build_commit` reads `unknown` and `build_number` reads `0` in `llama-bench`'s
own JSON.** The tree is staged from a pinned tarball, so it carries no `.git`,
and llama.cpp's build stamp is written from `git describe`. So the run is pinned
by two artifacts and not by a stamp: the **content manifest** of the 3,425
source files, and the **sha256 of every binary** the run linked. The chain from
the upstream revision to the bytes that decoded has three links, **two of them
closed and one open**, and the open one is named below rather than passed over.

1. **Upstream commit to content.** `git archive` of
   `10bf611e533d81f739128304991c5e133c6aebd8` from a clone whose `origin` is
   `https://github.com/ggml-org/llama.cpp` yields 3,425 files whose manifest is
   `56c26d15…f3cda`, and the tag `b10451` resolves to that same commit. The
   manifest is a pure function of the commit, so a reader can redo it:

   ```sh
   git archive 10bf611e533d81f739128304991c5e133c6aebd8 | tar -x -C "$d"
   ( cd "$d" && find . -type f -print0 | LC_ALL=C sort -z \
       | xargs -0 sha256sum | sha256sum )
   ```

2. **Content to the compiled tree. THIS LINK IS OPEN, and nothing here closes
   it.** No compiler ran in this lease. `job.log:44` reads
   `llamacpp_build=ALREADY PRESENT`; there is no `llamacpp_build_rc`, no build
   log, and the step 4 and step 5 headers are one second apart. The binaries
   were carried from the **2026-09-01 lease**, which recorded the identical
   `llama-bench` and `llama-cli` sha256s
   ([`rocm-strix-qwen38-q4km-20260901.md`](rocm-strix-qwen38-q4km-20260901.md)),
   and whose source reached the box as a `git archive` tarball with sha256
   `9f92a61d…54fb72`. A tarball's sha256 is not a pure function of a commit —
   it varies with the git version and the archive options — so it identifies
   those bytes without deriving them from `10bf611e…aebd8`. **No committed
   artifact ties the verified source of link 1 to the executed bytes of link
   3.**

   What the job did check, twice, is that the source tree sitting in the pod
   still hashed to `56c26d15…f3cda` (`job.log:38` and `job.log:53`). That is a
   true fact about that tree. It is not a fact about what any compiler consumed,
   because in this lease no compiler consumed anything.

3. **Compiled tree to the executed bytes.** Every binary and shared object the
   run linked is sha256'd in the table above, so what decoded is pinned as
   bytes.

**Link 1 was not checked by the job either. It is checked here instead, and it
was open until it was.** The job compared the staged tree against a constant
written into `job-as-run.sh`, which proves the tree did not change between two
points in the job and says nothing about which upstream revision that tree is;
the revision itself came from a `.VERSION` text file staged beside the tarball on
the share, which is an assertion rather than a check. Reproducing the manifest
from upstream is what makes the difference. A stamp would have recorded what the
tree claimed to be; the manifest records what it was.

**So the honest summary is three sentences.** The source is pinned to upstream
and verified, by a manifest that is a pure function of the commit. The executed
binaries are pinned as bytes, and carried from the named 2026-09-01 lease. The
step between them is asserted and not checked, and closing it needs a build in a
lease that records its own compiler invocation — which this row does not owe and
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) carries.

### `use_extra_bufts`, answered by observation

`llama-bench` exposes no switch for it. `src/llama-model.cpp:2489` defaults it
`true` and `tools/llama-bench/llama-bench.cpp:1218-1267` sets no override, so
this run's value is `true`. Rather than assert that, the run printed where the
tensors landed.

**This is an ASSEMBLED EXCERPT, not one transcript, and the provenance is per
line.** `llama-cli -v` allocates twice — a reserve pass and then the real load —
so the six buffer lines below do not occur together anywhere. Each is quoted
with its line in
[`system-info-extract.txt`](rocm-strix-llamacpp-denominator-20260902/system-info-extract.txt):

```
189  load_tensors:        ROCm0 model buffer size = 15345.66 MiB   <- real load
190  load_tensors:    ROCm_Host model buffer size =   682.03 MiB   <- real load
192  llama_kv_cache:      ROCm0 KV buffer size = 16384.00 MiB      <- real load
174  llama_memory_recurrent:  ROCm0 RS buffer size =   149.62 MiB  <- reserve pass
180  sched_reserve:      ROCm0 compute buffer size =   122.00 MiB  <- reserve pass
181  sched_reserve:  ROCm_Host compute buffer size =   266.02 MiB  <- reserve pass
```

The reserve pass's own `load_tensors:` lines read `0.00 MiB` at 87-88 and its
`llama_kv_cache:` line reads `0.00 MiB` at 154, which is why the first three are
taken from the second pass and not from beside the other three.

**The conclusion does not depend on the assembly.** It is an absence over the
whole file, so it is checked over the whole file rather than over this excerpt:
**no `CPU_REPACK` buffer type appears anywhere in the run's stderr**, in either
pass. With `-ngl 99` every loaded tensor sits in a ROCm buffer, so the CPU repack
buffer types that `use_extra_bufts` admits hold nothing and the flag cannot move
this measurement. `REPACK = 1` in `system_info` reports a compiled-in CPU
capability, not a buffer this run used.

### The model as the oracle read it

`866 tensors`, `n_layer = 64`, `n_layer_all = 65`, `qwen35.nextn_predict_layers
= 1`. Types: `f32` 456, `q4_K` 294, `q6_K` 67, `q5_K` 48, `q8_0` 1. The 65th
block is the MTP/drafter layer; `n_layer = 64` is what the trunk runs per token.

## Artifact

`unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10`,
`Qwen3.8-27B-Q4_K_M.gguf`.

| | |
|---|---|
| Bytes | 17,106,775,008, matched against the expected value |
| sha256 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| Verified | **on the worker**, against the worker-local `/tmp` copy, before any timing ran |
| On mismatch | the job aborts, rather than reporting the mismatch and continuing |

The `UD` family is deliberately not used: our loader could not read it when
#2497 was opened ([#2510](https://github.com/mudler/vllm.cpp/issues/2510)) and
the published `UD-Q4_K_XL` bytes have moved in place under an unchanged name, so
a repo id there is not a pin. The 12.616 tok/s harness-sanity figure the
predecessor took on `UD-Q4_K_M` is a **different artifact** and is context, not
this row's number.

## Environment and contention

| | |
|---|---|
| Device | `strix:gpu0`, Radeon 8060S Graphics, `gfx1151` (`0x1151`), 65,536 MiB VRAM |
| Host | AMD RYZEN AI MAX+ 395, `x86_64`, `nproc` 32 |
| Worker | `rc-worker-lcjhd`, Linux `6.14.0-36-generic` `#36~24.04.1-Ubuntu` |
| ROCm | 7.2.4 |
| `boot_id` | `a5bc8128-f6ad-4767-8614-6923f88032e1`, **one value across all 6 legs** — no reboot split this run |
| Lease | `rc` job `ff18a029-cd10-42d1-a5f7-9129c1c8af09`, one lease, one job at a time |
| Fleet at submission | `strix:gpu0` **ready** and taken by this job alone; `dgx:gpu0` and `thor:gpu0` busy with unrelated jobs on **different boxes**, `orin:gpu0` ready. Recorded verbatim in [`rc-devices-before.txt`](rocm-strix-llamacpp-denominator-20260902/rc-devices-before.txt) |
| Window | legs ran 22:58:20Z to 23:00:52Z, job 22:52:51Z to 23:00:53Z, 2026-09-02 |
| `fold_rc` | `0`, and the fold's own `status` is `MEASURED` with an empty `reasons` list |
| `JOB_VERDICT` | `OK` **by the script's own definition**, `fold_rc` being 0. The `JOB_VERDICT=` line itself appears in no artifact: `job-as-run.sh` copies the log out one line before it echoes the verdict, so what is captured is `fold_rc=0` and the verdict is derived from it |

The GPU work ran inside the `rc` lease throughout. No `ssh` to the device, and
no file mutex, because on a fleet device the lease is the path and a second
mutex the fleet cannot see is the #777 failure.

`HSA_ENABLE_SDMA` is retired (#2511) and is set nowhere in the job.

## Clocks, and why the instrument names itself

**Every clock figure here is AD-HOC.** No in-tree harness samples AMD clock
state: `tools/bench/gpu_clock_state.py`, the one helper
`.agents/benchmarking.md` names, reads NVIDIA fields and does not run on this
board. [#2381](https://github.com/mudler/vllm.cpp/issues/2381) owns that gap.
The sampler here is carried beside the job, reads
`/sys/class/drm/card*/device/{gpu_busy_percent,pp_dpm_sclk}` at 4 Hz to
**worker-local** disk, and is recorded as ad-hoc rather than presented as a gate
reading. A 4 Hz flush against CIFS stalls the sampler and distorts the very
sample spacing the window is judged on, which is why it does not write there.

Two windows are reported because they answer different questions. The whole
window includes the 17 GiB model load, during which the GPU idles at 600 MHz;
that dominates the raw spread and says nothing about decode.

| Leg | compute-window sclk (MHz), `busy >= 90%` | n | whole-window sclk (MHz) | n |
|---|---|---|---|---|
| 1 | med 2662.0, mean 2668.7, 2578–2799, 8.3% | 56 | med 2619.5, mean 2064.0, 600–2799, 83.9% | 102 |
| 2 | med 2635.5, mean 2663.6, 2565–2835, 10.2% | 56 | med 2614.0, mean 2089.5, 602–2835, 85.4% | 97 |
| 3 | med 2644.0, mean 2670.2, 2584–2827, 9.2% | 56 | med 2629.5, mean 2118.7, 604–2827, 84.5% | 96 |
| 4 | med 2647.0, mean 2672.2, 2559–2849, 11.0% | 56 | med 2594.0, mean 1928.0, 600–2849, 86.7% | 110 |
| 5 | med 2642.5, mean 2659.8, 2571–2797, 8.6% | 56 | med 2628.0, mean 2075.9, 602–2797, 83.5% | 98 |
| 6 | med 2635.5, mean 2656.7, 2561–2815, 9.6% | 56 | med 2616.0, mean 2081.0, 602–2815, 84.6% | 97 |

Every compute window holds **exactly 56 samples**, which is 14 s at 4 Hz and is
the same busy stretch each time. The compute-window medians span 2635.5–2662.0
MHz, a 1.0% spread across the whole run, against a 0.303% spread in the
throughput — the board held one clock state for all six legs.

**`.agents/benchmarking.md`'s 5% within-run SM-clock spread ceiling is not
applied here.** It was calibrated on a datacenter part with persistence mode and
its own power budget. This APU shares one package power budget with 32 CPU
cores, and every compute window either run has ever sampled on it is above that
ceiling: **8.3–11.0%** across these six legs, and **8.2–12.0%** across the
predecessor's five (8.2, 9.5, 10.2, 10.7, 12.0 — its
[own table](rocm-strix-qwen38-q4km-20260901.md), whose 9.5–10.7 band is the
three llama.cpp legs and whose 8.2 and 12.0 are our arm and a harness control).
The two ranges are not the same range, and the earlier draft of this line said
they were. The argument is unchanged at the true numbers, and stronger: eleven
of eleven compute windows, on two engines, in two leases, on an idle box, sit
between 1.6x and 2.4x the ceiling. A rule no observation has ever satisfied is
not a rule this board has failed. An AMD rule needs its own calibration
and cannot inherit the NVIDIA number; that observation belongs to #2381 and is
not repaired here. The cross-arm clause of that section — that two arms' clock
medians and means must agree, because the offset lands in the ratio — has no
subject, because there is one arm and there is no ratio.

## Reliability

**6 of 6 legs returned `rc=0`.** No `GPU Hang`, no `HW Exception`, no
`Memory access fault` in any leg's stderr; the job greps each leg for all three.
The predecessor recorded our *own* arm hanging on 2 of 3 legs (#2511) while the
oracle completed 3 of 3; the oracle's 6 of 6 here is the same observation with a
larger denominator, and it says nothing about our arm, which did not run.

One step did time out, by design: step 6's `llama-cli -v` system-info leg exits
`124`. `llama-cli` at `b10451` is a TUI in front of an in-process server and it
never stops redrawing on a closed stdin even with `-no-cnv` set — a first
submission of this job wrote 3.9 GB to the share in four minutes that way, which
is the 24.9 GB incident again. Its stdout goes to `/dev/null` and everything the
step is for — `system_info`, the device list, the buffer types, the memory
breakdown — arrives on stderr, which terminates. That leg is not a timing leg
and contributes to no figure here.

## Recipe

The exact script that ran is
[`job-as-run.sh`](rocm-strix-llamacpp-denominator-20260902/job-as-run.sh),
with [`fold.py`](rocm-strix-llamacpp-denominator-20260902/fold.py) and
[`amd_clock_sample.py`](rocm-strix-llamacpp-denominator-20260902/amd_clock_sample.py)
beside it. The timing command, once per leg:

```sh
llama-bench -m Qwen3.8-27B-Q4_K_M.gguf -p 0 -n 64 -ngl 99 -r 3 -o json
```

run inside the lease under `podman` with `--device=/dev/kfd --device=/dev/dri`,
`LD_LIBRARY_PATH` covering `build-llamacpp/bin` (without it these binaries do
not find their `*-impl.so` and exit 127), the GGUF staged to worker-local `/tmp`
because `/workspace` is CIFS and is never a run surface, and stdin closed.

## Files

| File | What it is |
|---|---|
| `RESULT.json` | the fold's complete output: every leg, every identity field, both clock windows |
| `leg{1..6}.json` | `llama-bench`'s own JSON per leg, unmodified |
| `leg{1..6}.rc` | each leg's exit status |
| `clock-leg{1..6}.jsonl` | the raw 4 Hz sysfs samples |
| `job.log` | the job log: the artifact and manifest verification, the binary sha256 table above, the lease id, `llamacpp_build=ALREADY PRESENT` and `fold_rc`. **Those five are what only this file carries** — the `boot_id` is not one of them; it is also in `worker.txt`, in `RESULT.json` and in all six clock files. `.gitignore`'s repo-wide `*.log` matches this path, so it is committed with `git add -f` as the other evidence logs under this directory are — an unforced `git add` drops it silently |
| `system-info.err.gz` | the **complete** `llama-cli -v` stderr. Uncompressed it is 4,952 lines / 348,370 bytes with sha256 `9c677901…08ca` — that sha is of the CONTENT, not of the `.gz`, which is a transport wrapper and whose own digest depends on the compressor. Committed so the file below can be checked against something |
| `system-info-extract.txt` | a **hand-made** selection from it: 194 of its 2,522 unique lines. No step of `job-as-run.sh` writes this filename. What is machine-checkable is that it is a faithful selection — every line appears verbatim in the source, in source order, with the log's `<time> <level> ` prefix stripped and duplicates dropped keeping the first. The command below reproduces it byte-for-byte from `system-info.err.gz` |
| `rc-devices-before.txt` | `rc devices` at submission: the fleet contention state the design requires, quoted in "Environment and contention" above |
| `list-devices.txt`, `worker.txt` | device enumeration and worker identity |
| `job-as-run.sh`, `fold.py`, `amd_clock_sample.py` | the recipe |

```sh
gunzip -c system-info.err.gz \
  | sed -E 's/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ [A-Z] //' \
  | awk '!seen[$0]++' \
  | grep -Fxf system-info-extract.txt \
  | diff - system-info-extract.txt        # empty
```

That command verifies fidelity; it does not regenerate the selection, because
the selection was made by a person. The two arguments that rest on this file —
the absent `CPU_REPACK` buffer type and the tensor-type counts — are both
re-checkable against `system-info.err.gz` directly, and `CPU_REPACK` appears
**0 times** in all 4,952 lines of it.

## What this does not establish

- **Nothing about our arm.** It did not run. Its token gate fails and its speed
  axis is inadmissible until that changes.
- **No ROCm floor.** A floor is a comparison and this row has one side of one.
  `BACKEND-GATE-ROCM-LLAMACPP` stays `INVENTORIED`.
- **No gate.** This figure is evidence. CI cannot re-derive it, and no checker
  reads it.
- **Not that these binaries were compiled from this source.** Link 2 above is
  open. The source is pinned to upstream and the executed bytes are pinned as
  bytes, and no committed artifact joins them.
  [#2497](https://github.com/mudler/vllm.cpp/issues/2497) closes it on the
  retake, which cannot reuse these binaries.
