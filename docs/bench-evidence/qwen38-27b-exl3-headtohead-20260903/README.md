# `qwen38-27b-exl3-headtohead` — their engine and ours, one box, one client

## What this measures, and why it exists

Until 4 September 2026,
[`docs/benchmarks/qwen38-27b-exl3-gb10.md`](../../benchmarks/qwen38-27b-exl3-gb10.md)
led with 1.25x on the Qwen3.8-27B EXL3 pair. That ratio was **59.5 tok/s counted
decode-only** against the 47.5 "decode tok/s" the [Mia-AiLab card][card] quotes.
The same run of ours read **45.1 tok/s counted over whole-run wall time**. The
card defines "decode tok/s" nowhere. If it counts with the prefill in, we were at
0.95x, slower, and the page led with the favourable half of an open question.

Prose could not settle that, and the page said so. This harness settled it by
measurement instead: it runs **their engine and ours on one box, behind one
client, on one workload, interleaved**, and computes **both counting conventions
from the same timings**. With one client and one definition the convention
question disappears, whichever way the answer falls.

[card]: https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw

## Status: the run finished on 4 September 2026

The job ran to completion on `dgx:gpu0` under lease
`d32255f7-2004-432a-b656-dcaef50037a9`. All four legs completed 164 of 164
requests with no failures. `results.txt` and the four `*.clientlog` files here
are the job's own output, copied unedited off the share.

| leg | order | decode-only tok/s | whole-run tok/s | mean TTFT ms | output tokens | wall s |
|---|---|---|---|---|---|---|
| `THEIRS-A` | 1 | 44.63 | 32.54 | 1021.7 | 19,680 | 604.9 |
| `OURS-A` | 2 | 53.68 | 37.33 | 1062.8 | 20,992 | 562.3 |
| `THEIRS-B` | 3 | 45.01 | 33.99 | 886.6 | 19,680 | 579.1 |
| `OURS-B` | 4 | 53.57 | 37.36 | 1055.1 | 20,992 | 561.8 |

```text
VERDICT decode-only : ours 53.63 vs theirs 44.82 = 1.197x
VERDICT whole-run   : ours 37.35 vs theirs 33.26 = 1.123x
```

Their acceptance counter reads 15,238 accepted draft tokens over 19,680 output
tokens in leg A and 15,253 over 19,680 in leg B. Ours reads nothing, for the
reason recorded further down.

The per-request JSON records and the server logs stay on the share at
`/mnt/nas_share/rc/exl3-headtohead/out/`. They are 1.6 MB of per-token timings
and nothing on the benchmark page is derived from anything but the summaries
here.

The reading of these legs, the axes they leave open, and what no gate covers is
on the benchmark page under
[the head-to-head](../../benchmarks/qwen38-27b-exl3-gb10.md#the-head-to-head-their-engine-and-ours-behind-one-client).

## The pin is OURS, not theirs

Their card pins no engine revision. This harness pins one so the measurement is
reproducible:

| | |
|---|---|
| engine | `https://github.com/MiaAI-Lab/exllamav3` |
| revision | `63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed` |
| identity | clean worktree, `origin` as above, read 2026-09-03 |
| staged as | `exllamav3-fork.tar.gz`, sha256 `18b49d64e6a171bcbfd06bd02f139fc53189e04b5ff8f510a3afbd622dd372d4` |

**This pin is a choice made here. It is not a revision they published, and their
47.5 was not necessarily measured at it.** The fork is a squashed mirror of two
commits, so it carries no upstream history to place the number against.

## The aarch64 blocker that was expected does NOT apply to this fork

`.agents/oracles/exllamav3.md` records, from three `orin:gpu0` leases, that the
STOCK `turboderp-org/exllamav3` pin does not build on aarch64: 7 of 129
translation units fail on `immintrin.h`, `__builtin_cpu_supports` and
`__builtin_ia32_pause`, because `#ifdef __linux__` is used where the author meant
"x86". That record is about the stock pin and it stands.

**It does not transfer to this fork**, which carries an explicit aarch64 port.
Read at the pin, all seven of those files now carry real architecture guards:

```text
avx2_target.{h,cpp}, avx512_target.{h,cpp}   #if defined(__x86_64__) || ...
cpu/moe_handoff.cu:152-158                   #elif defined(__aarch64__)
parallel/all_reduce_cpu.cu:110-116           #elif defined(__aarch64__)
parallel/all_reduce_cpu_avx{2,512}.cpp       #elif defined(__aarch64__)
```

and `start.sh` sets `TORCH_CUDA_ARCH_LIST="12.0;12.1"` on aarch64 specifically
for GB10. The fork's own commit subject names the port.

**This is a source reading, not a build.** It says the expected blocker has been
addressed upstream of us; it does not say the fork compiles here. Only the queued
job can say that, and it is written to report a build or torch-wheel failure
verbatim rather than as a slow result, because "their engine does not install on
this hardware" is itself a publishable answer.

## The one client

`client.py` speaks OpenAI `/v1/chat/completions` with `stream=true` and does not
know which engine answers. It reports, per leg, from the same timings:

```text
decode-only rate = 1000 / mean(tpot),  tpot = (latency - ttft) / (n_out - 1)
whole-run rate   = sum(completion_tokens) / wall clock of the whole leg
```

`ignore_eos` is deliberately not sent, on either side. Both published protocols
let the model stop, and our bench's `ignore_eos = true` is one of the differences
that made the existing numbers non-comparable.

### Two instrument defects were found and fixed BEFORE the run

Both would have produced a plausible wrong number that ran **against their
engine**, which is the failure mode this repository keeps hitting.

1. **A chunk is not a token.** Their SSE wrapper buffers with `HOLD_BACK = 16`
   characters and emits arbitrary string slices. Counting SSE chunks would have
   divided by roughly a fifth of the real token count.
2. **Their reasoning tokens are in a different field.** Their server splits the
   model's `<think>` block into `reasoning_content` and only the tail into
   `content`; ours leaves the tags inline in `content`. A client that counted
   only `content` would have measured their TTFT to the **end** of the reasoning
   block and dropped every reasoning token from the rate.

The client now counts `content` and `reasoning_content` alike for TTFT, and takes
token counts from each engine's own `usage`.

### The client was validated against known ground truth

Against a mock emitting a 200 ms prefill then 50 ms/token, including a role-only
opening delta that must not read as a token:

| quantity | ground truth | measured |
|---|---|---|
| TTFT | 200 ms | 207.8 ms |
| median ITL | 50 ms | 50.1 ms |
| decode-only (pooled) | 20.0 tok/s | 19.94 tok/s |
| whole-run | 14.3 tok/s | 14.09 tok/s |

and again against a mock shaped like THEIR stream — reasoning slices, 4 tokens
per chunk, a terminal usage chunk — where TTFT reads 229 ms against the 200 ms
prefill (it reads about 830 ms with the defect above) and the token count comes
back as 20 rather than as the 5 chunks that carried it.

## The one harness adaptation, and its boundary

`serve_openai-usage.patch` touches **only** `tools/serve_openai.py`, their server
wrapper. It touches no file under `exllamav3/`, so no engine, kernel or sampling
behaviour changes.

Their streaming path already computes the prompt and completion token counts and
the `Job`'s `accepted_draft_tokens` (`exllamav3/generator/job.py:262,691`), and
their non-streaming path already returns the same numbers — the streaming path
simply drops them on the floor. The patch forwards them and emits the standard
OpenAI `stream_options.include_usage` terminal chunk, which is the exact shape
`vllm.cpp`'s own chat stream emits. Without it a streaming client has no token
count to divide by on their side.

The patch applies cleanly to the pristine pinned tarball and the result compiles.

## What the job builds, on both sides

The container has no CUDA toolkit, so the job installs `cuda-toolkit-13-0` from
NVIDIA's `ubuntu2404/sbsa` repository and repairs the `targets/<triple>/`
layout that both our CMake and torch's `cpp_extension` expect at
`$CUDA_HOME/{include,lib64}`.

Ours is built from the same pinned tree the published number came from
(`5649e07d`, staged as `src.tar.gz` and pinned by its sha256), with the same
recipe:

```sh
cmake -S <src> -B <build> -G Ninja -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_CUTLASS_FETCH=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_{C,CXX,CUDA}_COMPILER_LAUNCHER=ccache
cmake --build <build> -j 4 --target vllm-server
```

Theirs is built with `TORCH_CUDA_ARCH_LIST="12.0;12.1"` and `MAX_JOBS=4`, which
is what their own `start.sh` selects on aarch64. `-j 4` on both sides because
unconstrained parallelism has OOM-rebooted this box.

Nothing measured is ever executed off `/workspace`: it is CIFS, holds no
symlink, and a network filesystem in the load path is a confound. Weights, the
build, the venv and the binary all live in `/tmp`; only a rebuild cache and the
results are written back to the share.

## Matching, and what stays unmatched

Matched across both engines: the box (`dgx:gpu0`, GB10), the client, the
workload (real HumanEval, 164 problems, `max_tokens = 128`), `temperature = 0.6`,
`top_p = 0.95`, `top_k = 20`, `seed = 0`, concurrency 1, the chat endpoint, each
model's own chat template, thinking enabled, and no `ignore_eos`. Their sampling
defaults are what set `top_p` and `top_k`; ours are told to match them.

Not matched, and recorded rather than hidden:

- **Their recipe is theirs, verbatim from the card**: `-cq nvfp4 -cs 262144`, a
  256K context with an NVFP4 KV cache. Ours runs its own default cache. Each
  engine runs at its own published configuration, which is the honest shape, but
  it is not one configuration.
- **`VT_DFLASH_PAGED=0` on our side.** The shipped paged draft route faults
  eagerly ([#2274](https://github.com/mudler/vllm.cpp/issues/2274)). The
  published page already carries this as a limitation.
- **Acceptance is asymmetric, and the reason is a mirror gap.** Theirs is read
  from the `Job` counter the patch surfaces. Ours is **not exposed on the server
  path at all**: `spec_drafts_proposed()` and `spec_drafts_accepted()` are
  accessors on `GpuRunner` (`include/vllm/v1/worker/gpu/runner.h:269-270`),
  reachable from the bench client and from nothing else — not `vllm.h`, not
  `/metrics`, not any route. The client reports ours as absent rather than as
  zero, because zero would read as "the draft never fired".

  vLLM at the parity pin `5559679229` exports four of these and we export none:

  ```text
  vllm:spec_decode_num_accepted_tokens_total
  vllm:spec_decode_num_draft_tokens_total
  vllm:spec_decode_num_drafts
  vllm:spec_decode_num_accepted_tokens_per_pos
  ```

  That is a mirror gap rather than a limitation of this measurement, it is owed,
  and it is why an engine we wrote can be asked for its acceptance only through a
  benchmark binary. It is tracked as
  [#2770](https://github.com/mudler/vllm.cpp/issues/2770); this file only records
  that the gap is what forced the asymmetry above.

## These numbers will NOT reproduce the published 59.5

The published 59.5 / 45.1 came from `vllm-bench`, which sends a **raw completion
prompt** with no chat template and with `ignore_eos = true`. This harness sends a
**chat turn** through each model's own template with thinking enabled, and lets
the model stop. Those are different workloads, so our leg here is not expected to
land on 59.5, and a reader must not read a lower number as a regression.

What this harness produces is a new, internally consistent pair: two engines
measured the same way at the same moment. The ratio between them is the result.
The absolute values are comparable to each other and to nothing else.

## Interleaving

Legs run `THEIRS-A`, `OURS-A`, `THEIRS-B`, `OURS-B`. A sequential A/B measures
drift along with the arm, and this repository has a recorded case of one
unchanged binary reading 36.8 and 78.9 tok/s in the same session. Each server is
started and stopped around its own leg, and readiness is polled with a bounded
loop that checks the process is still alive, so a server that never comes up or
dies mid-run is reported as that rather than as a slow result.

## Files

| file | what it is |
|---|---|
| `client.py` | the one client, driving both engines identically |
| `job-as-run.sh` | the leased job, exactly as submitted |
| `serve_openai-usage.patch` | the harness adaptation, scoped to their server wrapper |
| `results.txt` | the job's own record: build recipes, checkpoint sha256 values, leg table, verdicts |
| `*.clientlog` | one per leg, ending in the client's full `CLIENT_RESULT` summary |

All three of the first files are byte-identical to what ran on the box.
