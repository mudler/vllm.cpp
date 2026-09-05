# What each `qwen4_exp` arm emits now that BOTH run vLLM's chunked GDN prefill, 4 September 2026

Wave ARMTOKENS of [`KERNEL-GDN-CHUNKED-MIRROR`](../../.agents/specs/gdn-chunked-mirror.md),
[#2858](https://github.com/mudler/vllm.cpp/issues/2858), under
[#2612](https://github.com/mudler/vllm.cpp/issues/2612).

**The one-line result. Neither sequence moved, and they still agree on five of
eight — while the layer-0 divergence between them fell 20x.**

| arm | token ids | agrees with CPU |
|---|---|---|
| `--device cpu`, production default | `11751 13 15767 411 2029 11 1092 369` | — |
| `--device cuda`, production default | `11751 13 15767 411 1928 11 628 567` | **5 of 8** (indices 0,1,2,3,5) |

Both are **byte-identical to the sequences measured before the default moved**
([PREFILLDIV](qwen4exp-cuda-prefill-divergence-20260902.md) §2,
[#2547](https://github.com/mudler/vllm.cpp/issues/2547)). The CPU arm did **not**
change. The registry comment predicting the ids "are EXPECTED TO DIFFER" is
falsified, and this file is what corrects it.

**That is not because the new arm is inert.** The chunked CPU arm runs, and §5
measures it: `VT_GDN_CHUNKED` moves decoder layer 0's Gated DeltaNet block output
by `3.702e-04` on the same binary, and it moves the CPU arm to within
`1.772e-05` of CUDA where the sequential arm sat `3.525e-04` away — a **19.9x**
reduction in the divergence #2547 opened. The ids did not follow it, in either
direction. This is exactly what the spec's `## Scope` says to expect: agreement
between two of our arms is an argmax over near-ties and is not a monotone
function of the distance between them.

**Nothing here is a token gate, and nothing here is a speed.** No oracle decoded
this prompt. Each arm is n=1.

## 1. Why this measurement is possible now, and was not before

[#2849](https://github.com/mudler/vllm.cpp/pull/2849) landed a chunked CPU GDN
prefill mirroring vLLM's bf16 intermediate placement and **made it the default
for bf16**. Before it the two arms ran two different *algorithms* — the CPU arm
the exact sequential recurrence, the CUDA arm the chunked WY decomposition. The
port wave deferred re-deriving the sequences because it needed the 67.564 GiB
artifact and a lease; the spec's `## Now` names it. Nobody had measured either
arm under the new default.

## 2. What was run, and on what

**Tree.** `b767ebda4e55122b1a5473b9aa4027da67f77b75` — `origin/main` `d2b1bda2b`
plus **one deletion**, §3. No instrument was patched in; the fingerprint used in
§5 is committed on `main`. Source tarball sha256
`9a4a2934652d4cc9cfe63511eb8fd043f945f4f977bd1ca74412277f569a45d1`; the job
refuses at `exit 11`/`exit 12` if the tarball digest or `HEAD_SHA` differs from
the value written into the script before the lease was taken.

**Host.** `thor:gpu0` under `rc`, pod `rc-worker-n8smh`, aarch64, NVIDIA Thor,
compute capability 11.0, driver 595.78, nvcc 13.0.88, built `sm_110`, 14 cpus,
122 GiB. Jobs `77476a46-465b-429f-bfd4-0a0ed81ec011` (the build that failed, §3),
`c31b56ca-2394-4846-8f3c-6da7cdbed2b5` (run 1) and
`417fb134-5e55-4e48-83a6-2e1c43f08a9b` (run 2). `thor` is where the 5-of-8 was
measured, which is why it is where this was.

`i8mm` is **PRESENT**, so `vt::cpu::QuantRepackActive()` is true by default and
the repack chain is LIVE. That is asserted on the box rather than assumed,
because it is the precondition the repack check in §4 needs to be a
discriminator rather than a no-op.

**Request.** `examples/vllm-server`, greedy, `--temperature 0`, `max_tokens=8`,
one sequence at a time, prompt `The capital of France is`,
`--block-size 16 --num-blocks 128 --max-model-len 256 --verbose`, **no
`CUDA_LAUNCH_BLOCKING`**. `logprobs:1` is set because that is how the prior run
*read* the ids — the `"token_id:N"` fallback of `BuildCompletionLogProbs` — and
keeping it keeps the conditions equal. One binary,
sha256 `1d129fa0ab96663bea8f50f715117596241a7f2f8ae77e877ba5853bb198792f`, for
every arm in both runs; run 2 asserts that digest rather than rebuilding.

**Artifact.** The released `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S, staged
worker-local, identity asserted **inside the lease on the bytes the server
opened**, before any arm ran and again in run 2:

```text
RESULT ARTIFACT VERIFIED: shard1 sha256=88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd staged_bytes=72546461650 shards=3
```

That is the digest [`docs/USAGE.md`](../USAGE.md)'s registry records for shard 1
of `unsloth/Qwen3.8-Flash-Next-GGUF` @ `8bdc666649440e9bdc97e16f3f75782c98478ff5`,
path `UD-IQ1_S`, and the digest PREFILLDIV asserted. The job exits 42 on a
mismatch rather than measuring a different file. Shards 2 and 3 were not
re-hashed; their digests are the registry's.

## 3. The first run produced no arm at all, because `main` did not compile

`origin/main` `d2b1bda2b` **cannot be built with `-DVLLM_CPP_CUDA=ON`**
([#2861](https://github.com/mudler/vllm.cpp/issues/2861)). `f2bda11e3` (#2849)
lifted the `VT_GDN_CHUNKED` read into `vt::GdnChunkedPrefillEnabled` and left the
CUDA-local wrapper behind with internal linkage and no caller; our CUDA flags
carry `-Werror=all-warnings`:

```text
cuda_gdn.cu(3265): error #177-D: function
  "vt::cuda::<unnamed>::ChunkedPrefillEnabled" was declared but never referenced
RESULT BUILD rc=1 wall=920s objects=549
```

The spec's `## Now` already said why it landed: *"No CUDA gate was run. D0
changed CUDA's f32 default and nothing on a GPU has executed since."* Nothing had
compiled that arm. **The measured tree is `main` plus the deletion of those nine
lines and nothing else**, and the job asserts the dead function is absent
(`dead_cuda_wrapper=0`) as a fourth source precondition.

## 4. Run 1 — the five arms, and the two controls

One build, one staged copy, five loads. Every arm returned `http=200` with
**8 ids**, none degenerate, `steps completed: 11`, and an empty first-error line.

| arm | env | ids | load | request |
|---|---|---|---|---|
| **CPU-DEFAULT** | *(production, none)* | `11751 13 15767 411 2029 11 1092 369` | 45 s | 8 s |
| **CUDA-DEFAULT** | *(production, none)* | `11751 13 15767 411 1928 11 628 567` | 30 s | 338 s |
| CPU-GDNSEQ | `VT_GDN_CHUNKED=0 VT_CPU_QUANT_REPACK=0` | `11751 13 15767 411 2029 11 1092 369` | 30 s | 7 s |
| CPU-REPACK0 | `VT_CPU_QUANT_REPACK=0` | `11751 13 15767 411 2029 11 1092 369` | 15 s | 6 s |
| CUDA-REPACK0 | `VT_CPU_QUANT_REPACK=0` | `11751 13 15767 411 1928 11 628 567` | 15 s | 275 s |

Decoded: CPU `" Paris. Given this fact, what is"`, CUDA
`" Paris. Given this information, can we"`.

**The positive control passes.** `VT_GDN_CHUNKED=0` — the exact configuration and
algorithm that produced the published sequence — reproduces it. Had it not, this
harness would not be measuring what the prior run measured and no arm here could
be quoted.

**The repack check passes.** `VT_CPU_QUANT_REPACK=0` is identical to the default
on a host where the chain is live, so the aarch64 repack that once put a NaN in
layer 0 is not perturbing this answer. `VT_CPU_QUANT_REPACK=0` is likewise
identical on CUDA, so the prior run's use of it is not what makes the two runs
comparable — the ids are the same with and without.

**And run 1 could not tell one thing from another.** All three CPU arms agreeing
*including* `VT_GDN_CHUNKED=0` has two explanations that are not the same
finding: the chunked arm runs and flips no argmax, or the chunked arm is **not
reached** and both settings run the sequential recurrence. The spec's T4 says a
flag whose two arms coincide reads as a pass either way, and no log line names
the arm — `--verbose` prints request stages, and neither branch of
`cpu_ops.cpp:2136` logs. Run 1 settles nothing here. Run 2 does.

## 5. Run 2 — the chunked CPU arm IS reached, by a committed instrument

`VT_Q4EXP_LAYER_FP=3` (`qwen4_exp_forward.cpp:95`, 15 taps) is committed on
`main`; it reads bytes a kernel already wrote and writes nothing back. Three arms
on **the same binary by digest**, with `VT_CPU_QUANT_REPACK=0` held on all three
so `VT_GDN_CHUNKED` is the only difference between the first two. Each printed
**1314** fingerprint lines and `taps=437` per step on all three steps — the same
counted property PREFILLDIV read, asserted so that an arm printing nothing and an
arm whose taps agree cannot look alike.

Step 0, `rel = |a-b| / max(|a|,|b|)` on `sum|x|`:

| tap | CPU-CHUNKED | CPU-SEQ | rel | CUDA | rel |
|---|---|---|---|---|---|
| `emb` | 37.242316 | 37.242316 | 0.000e+00 | 37.242316 | 0.000e+00 |
| `wide` | 148.969264 | 148.969264 | 0.000e+00 | 148.969264 | 0.000e+00 |
| `L00 in` | 148.969264 | 148.969264 | 0.000e+00 | 148.969264 | 0.000e+00 |
| `L00 ahc.mix` | 6517.5925 | 6517.5925 | 0.000e+00 | 6517.5925 | 0.000e+00 |
| `L00 ahc.inj` | 0.952026367 | 0.952026367 | 0.000e+00 | 0.952026367 | 0.000e+00 |
| **`L00 blk`** | **1074.05248** | **1073.65489** | **3.702e-04** | **1074.03345** | **1.772e-05** |
| `L00 s.attn` | 232.272312 | 232.225395 | 2.020e-04 | 232.270443 | 8.047e-06 |
| `L00 mhc.mix` | 3615.47142 | 3613.82031 | 4.567e-04 | 3615.62777 | 4.324e-05 |
| `L00 mhc.inj` | 2.92224121 | 2.92212772 | 3.884e-05 | 2.92224121 | 0.000e+00 |
| `L00 moe` | 389.936505 | 389.976283 | 1.020e-04 | 390.025768 | 2.289e-04 |
| `L00 s.mlp` | 392.790737 | 392.881629 | 2.313e-04 | 392.853498 | 1.598e-04 |
| `ple` | 232.508111 | 232.752729 | 1.051e-03 | 232.818457 | 1.333e-03 |
| `s.ple` | 568.716336 | 569.04265 | 5.734e-04 | 569.080593 | 6.401e-04 |
| `out` | 27981.5263 | 27964.6752 | 6.022e-04 | 28054.1436 | 2.588e-03 |

**The flag routes, and layer 0 is where.** The first tap that differs is
`L00 blk` — the Gated DeltaNet block output — in **both** comparisons, from an
input (`L00 in`, `ahc.mix`, `ahc.inj`) that is bit-identical on all three arms.
26 of 42 taps differ between the two CPU arms. The chunked CPU arm is therefore
reached and computing, and explanation (b) of §4 is dead.

### 5.1 `q`/`k`/`v` ARE bf16, and run 2 proves it without reading any source

Run 1's summary printed `FLAG ROUTES ON CPU ... NO -- either the flag is read and
ignored or q/k/v are not bf16`. **Both halves of that disjunction are false, and
the line is superseded.** It is computed from the eight ids alone, which is a
weaker instrument than the fingerprint, and it is committed in
`run1-results.txt` where a reader could mistake it for the answer. It is not.

The proof needs nothing but run 2's own numbers. `vt::GdnChunkedPrefillEnabled()`
— the `VT_GDN_CHUNKED` read — has **exactly one consumer in the tree**:

```c++
// src/vt/ops.cpp:2218
return q_dtype == DType::kBF16 && GdnChunkedPrefillEnabled();
```

If `q_dtype` were anything but bf16 the conjunction is false whatever the flag
says, and the flag could not change one bit of output. It changed `L00 blk` by
`3.702e-04` and 26 of 42 taps, on one binary by digest with `q`/`k`/`v` allocated
identically. **Therefore `q_dtype == kBF16`, as a consequence of the measurement
rather than an inference about it**, and the third possibility — that the arm is
reached, the flag routes, and the two arms simply land on the same eight argmaxes
— is what happened.

Two further readings agree. The fingerprint prints the runtime dtype of every
tap, and all 14 step-0 taps read `bf16` on all three arms. And statically,
`GdnActDType()` (`qwen3_5.cpp:3584`) is bf16 unless `VT_GDN_BF16` starts with
`0`, while `qwen4_exp_gguf_weights.cpp:206` hardcodes `torch_dtype = "bfloat16"`
rather than reading it from the GGUF.

**Run 1's `ARM ... DTYPE LINES: 0` field settled nothing and was never used.** It
greps the server log for a dtype line, and no such line exists to find:
`--verbose` prints request stages only, and neither branch of `cpu_ops.cpp:2136`
logs. A probe that produced no lines at all is not a measurement of the dtype,
and it is recorded here as a null probe so its zero cannot be read as one.

**Two independent corroborations that this is the same experiment as
PREFILLDIV.** The CPU-sequential `L00 blk` reads `1073.65489` and the CUDA one
`1074.03345`; PREFILLDIV, on a different tree two days earlier, read
`1073.65489` and `1074.03345` — byte-identical to the printed precision on both
arms. The new CPU-chunked value `1074.05248` is a third, distinct number between
them.

**The GDN divergence source is closed, and it is a 20x move.** Against CUDA, at
the taps the Gated DeltaNet feeds:

| tap | PREFILLDIV: CPU-sequential vs CUDA | **this run: CPU-chunked vs CUDA** | factor |
|---|---|---|---|
| `L00 blk` | 3.525e-04 | **1.772e-05** | **19.9x** |
| `L00 s.attn` | 1.939e-04 | **8.047e-06** | **24.1x** |
| `L00 mhc.mix` | 4.999e-04 | **4.324e-05** | **11.6x** |
| `L00 mhc.inj` | 3.884e-05 | **0.000e+00** | exact |

**And the second source is now the dominant one.** The same table's MoE taps move
the other way: `L00 moe` goes 1.269e-04 -> 2.289e-04 and `L00 s.mlp`
7.160e-05 -> 1.598e-04, from an input that is now 11.6x closer. That is
PREFILLDIV §3's unremoved residue, and with the GDN term gone it is what the
whole-model divergence is made of: `out` improves only 3.189e-03 -> 2.588e-03,
1.23x, against 20x at the block. **This is why the ids did not move.** The
largest divergence anywhere is at step 2's `out`: 1.471839e-02 between the two
CPU arms, 2.320338e-02 between CPU and CUDA.

## 6. What this does NOT establish

- **It is not a token gate.** No oracle decoded this prompt. vLLM cannot run a
  GGUF artifact, and the llama.cpp arm aborts in `build_delta_net_chunking`
  before loading a byte — the blocker PREFILLDIV §8 recorded.
- **No speed number.** The wall times in §4 are liveness, on n=1 legs, on a box
  whose contention was not characterised.
- **One prompt, one length.** `T = 5`, a single partial chunk at `BT = 64`. The
  cross-chunk state carry — the half of the algorithm the spec's `## Owed`
  already flags as unmeasured against a real oracle — is not exercised here, and
  it is the half the bf16 state snapshot lives in.
- **The MoE residue is named, not diagnosed.** §5 shows it is now dominant. It
  does not say what it is. That is PREFILLDIV's open item and stays open.
- **The recorded explanation for the 5-of-8 is confirmed, not revisited.** It is
  tempting to read unchanged ids as "the divergence attributed to the algorithm
  difference did not move when the algorithms were unified". The premise is false
  here: it moved 19.9x at the block. What did not move is the argmax.
  `.agents/specs/qwen4-exp-flash-next.md` already states the general form — token
  agreement between our two arms "is not monotone in the distance between them,
  because the decode is an argmax over near-ties", and a CPU-vs-CUDA
  token-exactness gate "is not well posed for this architecture at this
  precision". This run is a second, independent instance of exactly that, now in
  the direction that spec had not observed: PREFILLDIV saw a 332x *closer* arm
  agree on *fewer* ids; ARMTOKENS sees a 20x closer arm agree on the *same* ids.
  Neither direction is monotone, which is the claim.
- **Nothing is claimed about the other six published quants**, about
  `num_reqs > 1` (refused by name), or about ROCm, Vulkan and Tenstorrent — two
  of which still have no chunked arm.
- **The three CUDA gates the spec lists as unrun are still unrun.** §3 means they
  were unrunnable, not merely unrun; running them is not this wave.

## 7. Reproduction

`rc` jobs above on `thor:gpu0`, 03:28-04:37 UTC. The two job scripts, the two
`results.txt` files and the two tap diffs are committed beside this file in
[`qwen4exp-gdn-chunked-token-ids-20260904/`](qwen4exp-gdn-chunked-token-ids-20260904/).

```sh
vllm-server --model <shard1>.gguf --device cpu --host 127.0.0.1 --port 8171 \
    --block-size 16 --num-blocks 128 --max-model-len 256 \
    --served-model-name qwen4exp --verbose
curl -H 'Content-Type: application/json' \
     -d '{"model":"qwen4exp","prompt":"The capital of France is","max_tokens":8,"temperature":0,"logprobs":1}' \
     http://127.0.0.1:8171/v1/completions
```

`--block-size 16` because the CPU attention backend accepts only multiples of 16.
Swap `--device cuda` for the second row of the headline table, and prefix
`VT_GDN_CHUNKED=0` for the sequential arm.
