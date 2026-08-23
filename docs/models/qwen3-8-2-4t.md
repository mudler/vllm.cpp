# Qwen3.8 2.4T

Qwen3.8-2.4T-A95B is a 2.4-trillion-parameter mixture-of-experts model. At the
`UD-Q1_0` quantization the weights are 369.97 GiB, and this engine serves them
from a box with 119.631 GiB of memory. The checkpoint is about three times the
size of the machine it runs on.

It works because the routed-expert towers are never copied. They stay in the
GGUF file mapping and the forward reads the slices it needs, so only the dense
remainder becomes resident.

## Read this before you download 370 GiB

Steady decode is **11.05 s/token**. That is seconds per token, not tokens per
second. The first token arrives about 80 s after a 4.3-minute load.

This is a capacity result. It shows that the model answers on hardware that
cannot hold it. It is not an interactive speed, and no software change reaches
one, because [the floor is storage](#why-you-cannot-make-this-fast).

## What it takes

| You need | Value |
|---|---|
| Disk | 370 GiB of free local NVMe for the checkpoint |
| Memory | About 62 GiB resident after load, measured on a 119.631 GiB box |
| Device | `--device cpu`. Read [the CUDA arm](#the-cuda-arm-and-why-not-to-use-it) before you try `--device cuda` |
| Load time | 255 s to 271 s from a dropped page cache |
| First token | 79 s to 86 s |
| Steady decode | 11.05 s/token |
| Concurrency | One request. The capacity argument stops holding as concurrency rises |

Every figure comes from one host: `dgx:gpu0`, a DGX Spark GB10 with 119.631 GiB
of unified memory, reading the checkpoint from local NVMe. Different storage or
a different host changes them.

## Get the weights

| Field | Value |
|---|---|
| Repo and revision | `unsloth/Qwen3.8-2.4T-A95B-GGUF` @ `567d3e6ac26c5474b18311e619c04350fb9a5556` |
| Publisher | Unsloth, a third-party quantization rather than a first-party release |
| Arm | `UD-Q1_0`, which stores the expert towers at 1.1875 bits per weight |
| Files | `UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-000{01..10}-of-00010.gguf`, ten shards |
| Bytes | 397 256 393 248 over the ten files, that is 369.97 GiB |
| Tensor records | 1702, equal to the `split.tensors.count` the shards declare |
| sha256, shard 1 | `b7770552b2ac24e7334c917bc92e90e218e87cfe29484db65e62e8ef2a60334d` for `-00001-of-00010.gguf`, 10 943 264 B |
| sha256, shard 2 | `2765517f833c736338d3ab34354e1c10eb8d79e62325f998285b435e5cf03dcd` for `-00002-of-00010.gguf`, 48 759 636 544 B |

```sh
hf download unsloth/Qwen3.8-2.4T-A95B-GGUF \
  --revision 567d3e6ac26c5474b18311e619c04350fb9a5556 \
  --include "UD-Q1_0/*" \
  --local-dir ./qwen3.8-2.4t-a95b-gguf
```

The files land under a `UD-Q1_0/` subdirectory of `--local-dir`, because that is
where they live in the repo. Put the copy on **local NVMe**. A network
filesystem adds an uncontrolled variable in front of the expert reads that every
token makes.

**A repo id alone is not a pin**, because a quantized checkpoint gets
re-quantized in place under an unchanged name. Shard 1's digest was recomputed
from the mirrored copy. Shard 2's is the download manifest's, and its byte count
was recomputed.

**The shard count is part of every file name.** A GGUF split writes the total
into each member's name, so `-of-00008` and `-of-00010` name different files, and
the wrong one gives a file-not-found after a 370 GiB download. The recipes on
this page carried `-of-00008` until
[#1420](https://github.com/mudler/vllm.cpp/issues/1420) corrected them. Shard 1 holds no
tensors at all. It carries the metadata and the split declaration, so it is the
file that says what the other nine are.

## Run it

Point `--model` at **shard 1**, not at the directory. Given shard 1 the loader
finds its nine siblings from the `-NNNNN-of-MMMMM.gguf` naming and cross-checks
`split.count`. A directory sends the loader down the Hugging Face branch, which
fails on a missing `config.json` before it looks for a GGUF.

### With the container

```sh
docker run --rm -p 8899:8899 \
  -v "$PWD/qwen3.8-2.4t-a95b-gguf:/models:ro" \
  ghcr.io/mudler/vllm.cpp:main-cpu \
  --model /models/UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf \
  --offload-config '{"vllm_cpp":{"mmap":{"enabled":true,"prefault":false},
                                 "expert_stream":{"enabled":true,"slots":4000}}}' \
  --device cpu \
  --max-num-seqs 1 \
  --max-model-len 512 \
  --port 8899
```

The container runs as UID `1000`, so that user must be able to read the weights
under `/models`. [The container guide](../guides/container-images.md) covers the
lanes and the tags.

**What has been run in the container, and what has not.** The `main-cpu` image
accepts this document and reports it back at startup, checked on 22 August 2026
against digest
`sha256:7f88301ea282dad778748929e7aa6869d2418c8d295eef0e7900cca8310d06e5`:

```text
engine: weight residency (offload_config vllm_cpp): mmap=on prefault=off expert_stream=on expert_stream_slots=4000
```

The 370 GiB run itself has not been repeated inside a container. Every measured
figure on this page comes from a source build on `dgx:gpu0`.

### From a source build

A plain CPU build is enough. No CUDA is involved on this path.

```sh
cmake -S . -B build
cmake --build build -j
```

```sh
./build/examples/vllm-server \
  --model ./qwen3.8-2.4t-a95b-gguf/UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf \
  --offload-config '{"vllm_cpp":{"mmap":{"enabled":true,"prefault":false},
                                 "expert_stream":{"enabled":true,"slots":4000}}}' \
  --device cpu \
  --max-num-seqs 1 \
  --max-model-len 512 \
  --port 8899
```

### What each flag does

| Flag | Why it is there |
|---|---|
| `prefault: false` | The setting that decides whether this works. Pre-faulting is on by default and is right for a model that fits. For 335.62 GiB of expert towers that cannot fit, it reads the whole checkpoint to fill a page cache that cannot hold it |
| `mmap: true` | Confirms the default rather than enabling it. It is what makes the checkpoint fit, because an expert tower is borrowed from the file mapping and costs zero anonymous bytes |
| `expert_stream`, `slots: 4000` | Off by default. 4000 is the count the published decode figure was measured at. [8000 is worse, not better](#why-you-cannot-make-this-fast) |
| `--device cpu` | The only arm with a passing correctness gate. See [the CUDA arm](#the-cuda-arm-and-why-not-to-use-it) |
| `--max-num-seqs 1`, `--max-model-len 512` | Keep the KV cache out of the way. Nothing is batched at this speed |

The recorded runs set the equivalent environment variables instead of the config
document: `VT_GGUF_PREFAULT=0`, `VT_MOE_EXPERT_STREAM=1` and
`VT_MOE_EXPERT_STREAM_SLOTS=4000`. The two forms are the same switches, and a
variable beats a config field wherever both are set.

## Check that it answered

Check readiness against the model list rather than against the process:

```sh
curl -sf http://127.0.0.1:8899/v1/models
```

```sh
curl -s http://127.0.0.1:8899/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf",
       "prompt":"Q: What is the capital of France? A:","max_tokens":4}'
```

The 16 August 2026 run answered ` Paris. Q: What`. The four recorded streaming
runs drive a fixed prompt of token ids instead, and all four returned the same
32 ids, which detokenize to ` Paris. Paris is a city located in the northern
part of France, on the Seine River. It is the largest city in France and is
known for its iconic`.

The output is coherent. The one-bit encoding and the borrowed-tower path are
both faithful enough to serve.

## What the run costs

Two runs are recorded on `dgx:gpu0` with the page cache dropped before each one
(`ENG-EXPERT-STREAM-DEVICE` W0e, 18 and 19 August 2026,
[`.agents/benchmark-record.md`](../../.agents/benchmark-record.md)):

| Axis | Run 1 | Run 2 |
|---|---|---|
| Load | 271.1 s | 255.7 s |
| First token | 85.90 s | 79.09 s |
| Peak resident set | 86.5 GiB | 86.5 GiB |
| Peak swap | not sampled | 6 883 MiB |

Resident memory settles at about **62 GiB** of 119 GiB after the load, measured
at 62.45 GiB on the same-lease CPU control run. That agrees with what the
checkpoint's own tensor table predicts: 21.56 GiB of `attn_qkv` and 17.25 GiB of
`ssm_out` expanded to bf16, plus 5.81 GiB of embeddings and F32 norms, so
44.6 GiB before the KV cache and the runtime.

Steady decode, all on the same box:

| Arm | Steady decode |
|---|---|
| Streaming on, 4000 slots | **11.05 s/token** |
| Streaming on, 8000 slots | 39.98 and 45.40 s/token |
| Streaming off | 66.7 s/token |

The 4000-slot figure is the median over steps 4 to 32, with a min of 9.43 and a
max of 13.25. A second run of the same arm gives 11.22, which is 1.54% above it.
The streaming-off row **carries no ratio against the other two**, because it was
taken on a different source tree on a different date.

Do not quote a first-token time as a decode number. Token 1 carries the prefill
and the cold expert set. From token 2 onward you are watching steady state.

## Why 370 GiB fits in 119 GiB

The model does not fit because of streaming. It fits because of **borrowing**.

The GGUF mapping is read in place on the CPU path, so a routed-expert tower
costs no resident bytes. Only the dense remainder is copied. Expert streaming
sits on top of that: it keeps a fixed arena of recently used expert slices in
memory, so that part of each token's reads come from RAM instead of from NVMe.

[The expert streaming guide](../guides/expert-streaming.md) owns the mechanism,
the config schema, the precedence rule, the statistics line, and what each
device can serve.

## Why you cannot make this fast

The arithmetic is short and it decides everything. The first three rows are read
from the checkpoint's own metadata:

| Quantity | Value |
|---|---|
| Blocks (`qwen35moe.block_count`) | 93 |
| Experts routed per block, of `qwen35moe.expert_count` | 10 of 512 |
| Projections per routed expert | 3 |
| Expert slices per token | 2790 |
| Bytes per slice | 2 490 368, that is 2.375 MiB |
| Expert working set per token | **6.95 GB**, that is 6.47 GiB |
| Slots this recipe reserves | 4000, a 9.28 GiB arena |

That is a working set, not an I/O rate, because the arena serves part of it from
memory. The recorded 32-token run at 4000 slots counted 37 096 hits against
58 538 misses.

**The floor is storage, not software.** 6.95 GB at the roughly 5 GB/s an NVMe of
this class sustains is 1.39 s/token whatever the code does, which is 0.72 tok/s.
Reaching 3 tok/s would demand about 21 GB/s of expert bandwidth, so most of those
reads would have to come from memory instead. The arena holds 4000 slices against
the 2790 a token needs, under one and a half tokens of working set, and
top-10-of-512 routing does not give consecutive tokens enough reuse to close the
rest.

**If you need conversational speed from this model you need more memory or fewer
active parameters, not better software.**

A bigger arena is also not a free knob, and the reason is the page cache rather
than the arena. The same binary at 8000 slots measured a 39.98 to 45.40 s/token
median over two runs, and the second consumed all 30 625 MiB of the box's swap.
The extra 9.27 GiB of arena takes the free memory that the borrowed 370 GiB
expert mapping is served out of.

The arena is also measurably not what exhausted the box in
[#1299](https://github.com/mudler/vllm.cpp/issues/1299): a 64-slot 0.15 GiB arena
failed exactly where an 8000-slot 18.55 GiB one did, so this knob was never the
lever there either.

## What this does not establish

- **The quantization is extreme.** The expert towers hold 1.1875 bits per
  weight, and they are about 97% of the parameters. The output is coherent. This
  is not the configuration to judge the model's quality by.
- **There is no oracle.** `UD-Q1_0` stores its expert towers as `IQ1_XXXS`,
  which upstream llama.cpp does not define. The encoding exists only in the
  `unslothai/llama.cpp` fork, pinned in
  [`.agents/oracles/llama-cpp-unsloth.md`](../../.agents/oracles/llama-cpp-unsloth.md)
  and recorded `gateable = no`, because it has not been shown to build and run
  this model. [#933](https://github.com/mudler/vllm.cpp/issues/933) owes that
  measurement. So there is no token-exact and no throughput denominator, and
  every figure here is an absolute measurement compared against nothing.
- **One request at a time.** Nothing here says anything about concurrency.
- **One box.** Every number was taken on one DGX Spark GB10 with the checkpoint
  on local NVMe.
- **No `--device cuda` number.** [`docs/BENCHMARKS.md`](../BENCHMARKS.md) carries
  G0-SPEED as `VOID`, because a speed number behind a failing correctness gate is
  not a result.

## The CUDA arm, and why not to use it

`--device cuda` loads and decodes this checkpoint on a probed integrated part:
32 of 32 steps, at peak RSS 97.75 GiB of a 119.631 GiB box. **Its token gate
against the CPU arm does not pass**, so every number on this page was measured on
the CPU arm.

The 32 ids match the CPU arm for eight tokens and diverge at the ninth, ` the`
against ` France`. The margin there is very small: the CPU arm's own
second-ranked token is exactly the one the CUDA arm emitted, behind by 0.1% of
the winning logit. The CUDA continuation then falls into a mechanical recursion,
which is NOT evidence that the CUDA arm is the wrong one: prefilled down the
same branch, the CPU arm recurses in exactly the same way (2026-08-23,
[#1783](https://github.com/mudler/vllm.cpp/issues/1783)). **What causes the
divergence is not identified.**

This paragraph said "six tokens", "the seventh" and "both continuations are
coherent" until 2026-08-23. All three came from a transcription error in the
agent-facing benchmark record, which dropped one token id twice.

The host-weight alias is excluded, measured on
GB10 with bit-identical output from a `cudaMalloc` operand and from a 256-aligned
host one, but excluding one cause is not identifying another.

**On 21 August 2026 a router dump moved the failure upstream of the sampler.** On
a later tree, source `cffe59b`, the two arms already select different MoE experts
in the first block of the first forward, eight tokens before any emitted token
differs, and they differ there in the router GEMM input rather than in anything
the router does with it.

Three probes narrow it further, and none of them names the cause:

- **The router gate weights are excluded by measurement.** Their fingerprint is
  identical on both arms.
- **The two top-k implementations agree** with a plain lowest-index-wins rank of
  each arm's own logits. That held on only 5 of the 552 token-rows the dumps
  hold, so read it as a sample and not as a property of either implementation.
- **The embedding output is bit-identical.** A third probe dumped the hidden
  state before any GEMM, norm, or attention touches it, and 0 of 40,960 bf16
  values differ.

So the weights are the same at both ends of the stack, and the divergence starts
in the compute inside the first block. The cause is still not identified, because
the expert projections, the attention weights, and the norms were never
fingerprinted.

This passage used to end "The CUDA continuation also degenerates into a mechanical
recursion after the tokens the two arms share, which a coin flip between two
equally good tokens does not produce." A coin flip does produce it. On 2026-08-23
the CPU arm, prefilled down the same branch, recursed in exactly the same way
([#1783](https://github.com/mudler/vllm.cpp/issues/1783)), so the recursion belongs
to the branch and not to the arm that took it, and it says nothing about which arm
is wrong. The degeneration is still observed; only the inference from it is
withdrawn.

Five further conditions gate the arm, and all of them must hold at once.

- **The device must be probed capable, and most are not.** The condition is
  `cudaDevAttrPageableMemoryAccess AND cudaDevAttrIntegrated`, an integrated
  unified part. A discrete card answers false, keeps staging every tower, and
  keeps the refusal.
- **The model must be one whose forward reads experts through the slot seam.**
  Today that is the Qwen3.5 MoE family, `Qwen3_5MoeForConditionalGeneration` and
  `Qwen3_5MoeForCausalLM`, which a `qwen35moe` GGUF resolves to. Every other
  architecture keeps the whole bound and keeps the refusal even with
  `VT_MOE_EXPERT_STREAM=1` set.
- **The expert towers must keep the form the file stores them in**, which means
  keep-quant or keep-f16. `VT_GGUF_KEEP_QUANT=0` and an NVFP4 GGUF both keep the
  refusal. A file that mixes a kept tower with a staged one keeps it as well
  ([#1378](https://github.com/mudler/vllm.cpp/issues/1378)).
- **`VT_QWEN35_ALIAS_HOST_WEIGHTS` must stay on**, which is the default. It hands
  the kernels the host bytes directly instead of keeping the dense weights
  resident twice, and it is what makes the arm decode at all. Set it to `0` for
  the same-binary A/B back to the staging behavior.
- **No speed claim is attached.** Device access to host-resident weights on that
  part has a recorded penalty, and this lane reads about 6.95 GB of expert bytes
  per token that way, so a CUDA arm slower than the CPU arm remains a real
  possible outcome. No published figure bounds this either way.

### When CUDA refuses at load

`--device cuda` refuses at load when the weights cannot be staged into device
memory ([#1123](https://github.com/mudler/vllm.cpp/issues/1123)). For this
checkpoint that is 276 towers of 1 275 068 416 bytes plus three of
2 818 572 288, so 335.62 GiB against a pool `cudaMemGetInfo` reports as
128 452 956 160 bytes (119.631 GiB). The message names the byte counts on both
sides and what is missing.

It used to load for 26 minutes, report ready, and then die on the first request
with `vt cuda: cudaMalloc: out of memory`.

[The expert streaming guide](../guides/expert-streaming.md#what-each-device-can-serve)
covers what the bound counts, what it deliberately does not count, and how to
move it. Size it with `cudaMemGetInfo` and not with `nvidia-smi`, which answers
`[N/A]` on a GB10 because host and device share one pool.

## Tuning

This recipe sets four knobs, and
[the expert streaming guide](../guides/expert-streaming.md) owns the complete
schema, the precedence rule, the config-parse errors, and the statistics line
that tells you whether the lane ran.

Two things from it are worth repeating here, because a reader of this page hits
both. **Precedence is environment variable, then config, then built-in default**,
so an exported `VT_MOE_EXPERT_STREAM=0` beats a config `"enabled": true`. And a
misspelled key is refused at startup rather than ignored, so a typo cannot
silently leave the weights unborrowed.

## Where the rest lives

| You want | Read |
|---|---|
| The streaming mechanism and its statistics | [expert streaming guide](../guides/expert-streaming.md) |
| Every server flag and endpoint | [server reference](../reference/server.md) |
| The complete measurement record | [`docs/BENCHMARKS.md`](../BENCHMARKS.md) |
| Container lanes and tags | [container images](../guides/container-images.md) |
| Design rationale and evidence | [`.agents/specs/expert-streaming.md`](../../.agents/specs/expert-streaming.md) |
