# The `-ngl 99` split EXPLAINED: it is one tensor, and it is a kernel refusal llama.cpp has and we do not

Row `MODEL-MM-QWEN4-EXP`. Issue
[`ISSUE-LOCAL-01M2DAFPJN13SKQDF3NAS849H2`](../../.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2DAFPJN13SKQDF3NAS849H2.md).
Oracle [`llama-cpp-qwen4exp`](../../.agents/oracles/llama-cpp-qwen4exp.md), pin
`035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`. Board `strix:gpu0`, `gfx1151`,
reached by `rc run -d`, never `ssh`.

This discharges the last UNVERIFIED line of
[`qwen4exp-llamacpp-denominator-gfx1151-20260913.md`](qwen4exp-llamacpp-denominator-gfx1151-20260913.md):
*"The `-ngl 99` CPU/GPU buffer split is measured but not EXPLAINED. Which
tensors llama.cpp routes to the CPU buffer, and why that is 13x faster than full
`ROCm0` residency on a unified-memory part, is not established here."*

**No ratio appears here.** Every figure is llama.cpp measured against itself, in
one lease, on one binary, on one artifact. The `qwen4_exp` ROCm arm still has no
declared token-exact gate
([`ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG`](../../.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG.md)),
so no vllm.cpp number is divided by any llama.cpp number, and no vllm.cpp leg
was run.

## The answer

**Which tensors.** Exactly **one**. The 27,465.95 MiB CPU model buffer is
`per_layer_token_embd.weight`, an `IQ4_NL` per-layer-embedding (PLE) n-gram
table, and nothing else. This is read off the loader's own `-v` output, not
inferred:

```
tensor per_layer_token_embd.weight (27465 MiB iq4_nl) buffer type overridden to ROCm0
done_getting_tensors: tensor 'per_layer_token_embd.weight' (iq4_nl) (and 0 others)
    cannot be used with preferred buffer type ROCm_Host, using CPU instead
```

`(and 0 others)` is llama.cpp counting for us. The arithmetic closes
independently: `IQ4_NL` is 32 elements in 18 bytes and this tensor's row is
`ne[0] = 160` (`qwen4exp.embedding_length_per_layer_input = 160`), so a row is
`160/32 x 18 = 90` bytes, and `27465.95 MiB / 90 B` is 320,001,511 rows against
a `ple.head_vocab_sizes` sum of 16 heads of just over 20,000,000 each.

**Why it is 13x.** Not placement, not bandwidth, not the expert kernel. The HIP
backend **refuses `GET_ROWS` on this tensor**, at
`ggml/src/ggml-cuda/ggml-cuda.cu:5012-5016` of the pin:

```c
case GGML_TYPE_IQ4_NL:
case GGML_TYPE_MXFP4:
    // 32-value sub-blocks, the row size does not guarantee
    // the QK_K super-blocks the get_rows kernel iterates on
    return op->src[0]->ne[0] % QK_K == 0;
```

`QK_K` is 256 (`ggml/src/ggml-common.h:89`) and this row is 160, so
`160 % 256 = 160` and `supports_op` returns **false**. That single predicate
produces both halves of the observed behaviour:

1. It is why the *preferred* buffer type `ROCm_Host` is rejected at load and the
   tensor lands in a plain `CPU` buffer, where the CPU `GET_ROWS` reads it in
   place at zero copy cost. **The default split is a workaround, and it works.**
2. It is why forcing the tensor into `ROCm0` is catastrophic.
   `ggml_backend_sched` cannot run that `GET_ROWS` on ROCm, assigns it to the
   CPU backend, and at `ggml/src/ggml-backend.cpp:1381-1389` calls
   `ggml_dup_tensor_layout` to make a host-side copy of the **whole** source
   tensor, marked input *and* output so `ggml-alloc` may not reuse it. That
   copy is re-made on **every graph evaluation, that is, every decoded token.**

The scheduler prints the size of that copy and it is the measurement:

| probe | `ROCm_Host` COMPUTE buffer | `tok/s` |
|---|---|---|
| `-ngl 99` default | **1.29 MiB** | 25.4671 |
| `-ot per_layer_token_embd=ROCm0` | **27,467.24 MiB** | 1.9988 |

27,467.24 MiB is 27,465.95 MiB (the tensor) plus the 1.29 MiB the graph needed
anyway. The extra step cost implied is
`1/1.9988 - 1/25.4671 = 461.0 ms`, and `28.80 GB / 0.4610 s = 62.5 GB/s`, an
ordinary device-to-host copy rate for this part. **The mechanism, the byte
count and the wall clock agree.**

## The discriminating measurement

`llama-bench` at the pin, `-p 0 -n 64 -ngl 99 -r 3 -v`, one leg (one model load,
llama.cpp's own warmup, 3 timed generations of 64 tokens) per probe, same
binary `03bc5011…a1597`, same artifact `88a14208…3bafd`, same lease, same
`boot_id 2ed44130-3e9a-487f-876c-5cbb57f38aab` as the 12-leg denominator run.

| probe | `-ot` argument | `ROCm0` model buf | `CPU` model buf | `ROCm_Host` COMPUTE buf | overrides applied | `avg_ts` | the 3 repetitions |
|---|---|---|---|---|---|---|---|
| `A_ctl` | *(none, the production default)* | 41,368.28 MiB | **27,465.95 MiB** | **1.29 MiB** | 0 | **25.4671** | 25.6417 / 26.2547 / 24.5047 |
| `Q[0]` | `per_layer_token_embd=CPU` | 41,368.28 MiB | 27,465.95 MiB | **1.29 MiB** | 1 | **26.0700** | 26.1081 / 26.5073 / 25.5947 |
| `P_ple_dev` | `per_layer_token_embd=ROCm0` | **68,834.23 MiB** | *absent* | **27,467.24 MiB** | **1** | **1.9988** | 1.98559 / 2.02227 / 1.98866 |
| `C_all_dev` | `.*=ROCm0` | 69,175.25 MiB | *absent* | **27,467.24 MiB** | **1224** | **1.8813** | 1.82702 / 1.88847 / 1.92835 |
| `Q[1]` | `.*=ROCm0` *(second variant of the same probe)* | 69,175.25 MiB | *absent* | **27,467.24 MiB** | 1224 | **1.9575** | 1.84599 / 1.98781 / 2.03884 |
| `R_exps_host` | `ffn_(gate\|up\|down\|gate_up)_exps=CPU` | **3,368.27 MiB** | 27,465.95 + **38,341.02 MiB** | — | 144 | **OOM, no leg** | — |

**Three independent reproductions of the collapse** — `P` at 1.9988, `C` at
1.8813, `Q[1]` at 1.9575 — against three of the default at 25.4671, 26.0700 and
(from the landed 12-leg population) 25.877. The two states do not overlap and
are not close to overlapping.

**`P` is the whole finding.** It moves **one** tensor and leaves 1223 exactly
where the default put them, and it reproduces `C`'s collapse to within the
6.2% that separates `C` from `P`. `C` moves all 1224 and buys nothing further.
Between them they bracket the term: **the 13x is that tensor and only that
tensor.**

`Q[0]` is the explicit control for the default: naming the tensor's existing
placement changes nothing, and it lands inside `A`'s own spread. It is
`Q[0]` and not `Q` because `llama-bench` treats a repeated `-ot` flag as a
separate benchmark VARIANT rather than as a second override in one
configuration (`tools/llama-bench/llama-bench.cpp:925-947`: `,` separates
variant groups, `;` separates overrides inside a group), so the probe written
as `-ot per_layer_token_embd=CPU -ot ".*=ROCm0"` ran two legs instead of one.
Its second leg reserved the same 27,467.24 MiB host compute buffer. **The
harness defect is reported rather than hidden**, and the combined placement is
re-run with the `;` form in the follow-up lease below.

The implied copy rate agrees between the two independent pairs:

| pair | step delta | 28.80 GB at that delta |
|---|---|---|
| `A` (39.27 ms) vs `P` (500.30 ms) | 461.03 ms | **62.5 GB/s** |
| `Q[0]` (38.36 ms) vs `C` (531.55 ms) | 493.19 ms | **58.4 GB/s** |

Two placements, two collapses, one copy rate. **That is the discriminating
measurement**, and it is a rate a device-to-host copy on this part can produce
while no plausible page-migration or bandwidth story predicts the same number
twice from the same byte count.

### `R` produced no leg, and the reason IS the H1 answer

`R` moved the 144 routed-expert tensors to the host to test H1 directly. It
never reached a timed generation — the job died during `load_all_data` and its
`.json` is 0 bytes — but its stderr was salvaged and records exactly why:

```
tensor blk.47.ffn_gate_exps.weight (206 MiB iq2_xxs) buffer type overridden to ROCm_Host
ggml_cuda_host_malloc: failed to allocate 38341.02 MiB of pinned memory: out of memory
load_tensors:          CPU model buffer size = 27465.95 MiB
load_tensors:        ROCm0 model buffer size =  3368.27 MiB
load_tensors:          CPU model buffer size = 38341.02 MiB
```

The expert towers are **38,341.02 MiB** and this worker has **30 GiB of host
RAM** (`free -g`: total 30, available 27). The pinned allocation was refused, the
plain CPU buffer was taken instead, and the load was then killed filling it. So
on this board the "second compute engine" H1 describes **cannot be handed the
weights to compute with at all** — not slowly, not at a penalty, but not at all.
`R` is quoted whole, including its failure, because the failure is the
measurement.

It also shows the default is not leaving free capacity on the table: with the
experts moved off, only 3,368.27 MiB remain on `ROCm0`, so in the default
configuration essentially the entire device-resident 41,368.28 MiB **is** the
expert towers.



### The follow-up lease could not run, and why

A second lease was submitted to re-run the combined placement with the `;` form
and the scaled-down H1 control. **It refused before it measured**, by its own
precondition:

```
FATAL: no /tmp/ckpt-iq1s/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
```

The 67.55 GiB artifact is no longer on the worker's local overlay. The pinned
llama.cpp build tree still is — the job located `llama-bench` and matched its
`sha256` before it reached the artifact check — so `/tmp` was not wiped; the
checkpoint specifically is gone. **The job refused rather than measuring a
substitute, which is the behaviour the precondition exists to produce.**
Re-staging it from the CIFS share is a lease of its own: the landed denominator
record measured 657 s of 696 s of a first load as page-cache read wait on that
share, which is why every timed leg reads a local copy.

So `Q2` (the true combined placement) and `R2` (the scaled H1 control) are
**UNVERIFIED** and owed. Neither is load-bearing for the finding: `P` already
isolates the term to one tensor with three repetitions, and `C` reproduces it
independently over all 1224.

## Which hypothesis survives

The investigation was framed around three hypotheses. **The measurement refutes
all three, and the surviving explanation was not among them.**

**H1 — shared-bandwidth parallelism (the hybrid split adds a second compute
engine for free). REFUTED, twice.** First, in the fast configuration the CPU
performs no matmul at all — it performs one `GET_ROWS` gather of a handful of
160-element rows per token, so there is no second compute engine to speak of.
Second, `R` shows that on this board there could not be one: the expert towers
are 38,341.02 MiB against 30 GiB of host RAM, so the host cannot hold the
weights a CPU compute arm would need. H1 also predicted about 2x and the
observed factor is 13x, which its own framing already flagged as insufficient.

**H2 — the `IQ1_S` path on `gfx1151` is slow on the GPU. REFUTED, and it was
already refuted by the landed evidence.** The 512-expert `IQ1_S` towers are on
`ROCm0` in the FAST configuration. The landed `B` probe (`-ot exps=ROCm0`)
matched 144 tensors and reproduced the baseline buffer split byte for byte,
because those tensors were already there. The `P` probe moves the experts not at
all and still collapses.

*(One correction to the landed record: that document says B's "`-ot` pattern did
not match". It did match — `n_overrides_applied=144` — and was a no-op because
the tensors were already device-resident. The document's conclusion, that B is a
valid control, is unaffected.)*

**H3 — residency thrashing / driver page migration. REFUTED.** The cost is
visible in llama.cpp's own scheduler accounting before a single token is
decoded, as a reserved 27,467.24 MiB host compute buffer. It is an explicit,
accounted, application-level copy, not an opaque driver migration. No
`svm_range_set_attr` sampling is needed to exclude it, and none was taken.

**H4 — the surviving explanation: a per-token full-tensor copy forced by an
unsupported-op fallback, triggered by a row length that is not a multiple of
`QK_K`.** Proven by three independent instruments that agree: the source
predicate, the scheduler's reserved buffer size, and the wall clock.

## This falsifies the premise of the investigation

The question was *"why is the hybrid split 13x faster than full-GPU residency,
so that we can port it deliberately"*. **There is nothing to port.**

- "Hybrid placement beats full residency on a unified-memory part" is **not**
  what was measured. What was measured is that ONE tensor cannot be gathered by
  llama.cpp's HIP kernel, so llama.cpp keeps it on the host. The 27.5 GiB is not
  a deliberate bandwidth split; it is the fallback arm of a capability check.
- **vllm.cpp already implements the capability llama.cpp is missing.**
  `EmbeddingQuantKernelRocm` (`src/vt/rocm/rocm_embedding_quant.hip:130`,
  kernel at `:51`, `IQ4_NL` instantiated at `:88`, registered at
  `src/vt/rocm/rocm_ops.hip:207`, BACKEND-ROCM-QUANT-GATHER
  [#3093](https://github.com/mudler/vllm.cpp/issues/3093)) computes
  `blocks_per_row = width / BlockElems(dtype)` (`:142`) and launches one thread
  per selected 32-element block. **It carries no `QK_K` super-block assumption
  at all**, so `ne0 = 160` is an ordinary 5-block row for it.
- So on this tensor the two engines are not making the same trade. llama.cpp is
  host-resident because it must be. vllm.cpp is device-resident because it can
  be.

`docs/FEATURES.md:112` already predicted this in general terms — *"its CUDA
`get_rows` dispatches the legacy quants only"* — and that sentence is too
strong: the pin's list at `ggml-cuda.cu:4986-5011` admits `Q2_K` through
`Q6_K`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M` and
`IQ4_XS`. The correct statement is narrower and sharper: **`IQ4_NL` and `MXFP4`
are admitted only when `ne0 % 256 == 0`, and this model's PLE row is 160.** The
conclusion of that FEATURES line stands; its reason is now measured rather than
asserted.

## What this does NOT establish

- **It says nothing about the gap between the two engines' decode speeds.** Our
  arm is not paying this cost: a 28.80 GB per-token copy would put us near
  2 tok/s, and the landed liveness reading is 5.0-5.3 tok/s
  (`docs/FEATURES.md:163`, nine samples over six launches, 5.8% spread). Where
  our step time goes is **UNVERIFIED and owed**, and this document does not
  guess.
- **No ranked kernel table for our arm was produced.** See "What was not run".
- **No token gate on either side**, unchanged from the landed denominator.

## What was NOT run, and is UNVERIFIED

- **A ranked kernel table for the vllm.cpp ROCm arm on `gfx1151`.** We own no
  per-kernel timing instrument on ROCm. `VT_OP_PROVIDER_STATS` counts provider
  selections and reference-tier fallbacks, not time
  (`src/vt/op_provider.cpp:157,165`; `include/vt/op_provider.h:181-189`), and
  `src/vt/rocm/rocm_backend.hip:460` records that a `rocprofiler` equivalent of
  `VT_CUDA_PROFILE` is owed. The out-of-process route is `rocprofv3`
  (`tools/bench/strix_kernel_trace/worker.py:335`), and the last session to run
  it on this board recorded **invalid profiler timestamps that prevented kernel
  timing attribution**
  ([`strix-kernel-trace-3015-20260907/README.md`](strix-kernel-trace-3015-20260907/README.md),
  [#3040](https://github.com/mudler/vllm.cpp/issues/3040)). Producing that table
  needs #3040 closed first, plus a `gfx1151` build of this tree and a decode-only
  window; it is a wave of its own and is owed by the issue this document files.
- **The combined `per_layer_token_embd=CPU;.*=ROCm0` placement**, which would
  show directly that full device residency of the other 1223 tensors is not
  slow. `Q[0]` and `C` together already imply it; the direct leg is owed.
- **A TIMED H1 control.** `R` established that the host cannot hold the experts,
  which refutes H1 on this board, but it produced no `tok/s`. The scaled 8-layer
  retry that would have produced one never got an artifact. **H1 is refuted by a
  capacity fact and by the mechanism, not by a timed control leg**, and that
  distinction is recorded rather than smoothed over.
- **An expert-offload monotonicity sweep.** It was designed and is now
  pointless: the term is a single tensor, so there is no fraction to sweep.
- **Any page-migration, `wchan` or `mem_info_vram_used` sampling during decode.**
  H3 is excluded by the scheduler's own accounting, so the sampling would have
  measured a hypothesis that is already dead.
- **A decode-only clock window.** Unchanged from the landed denominator: a leg
  is ~46 s of which ~35 s is model load, the clock cannot be pinned inside a
  lease ([#1354](https://github.com/mudler/vllm.cpp/issues/1354)), and the
  pooled SM-clock spread on this board was 39.106%. **A within-engine delta
  below about 10% is not established by this instrument.** Every delta reported
  above is 12x or larger, so the instrument is adequate for the claim made and
  for no smaller one.

## Reproducing it

```sh
rc run -d strix:gpu0 --max-runtime 75m -- bash -c "$(cat ple-job.sh)"
```

`ple-job.sh` is committed beside this file, as run. It asserts the `llama-bench`
and artifact `sha256` before it measures and exits non-zero rather than
measuring a mismatch.
