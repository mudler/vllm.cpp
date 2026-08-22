# Qwen3.8 2.4T

Use this page for Qwen3.8 2.4T checkpoints, commands, supported arms, and current limitations.

## Which device can serve it

`--device cpu` serves this today, and that is the arm every published number for
this checkpoint was measured on. `--device cuda` now decodes it on a probed
integrated part; see the six limits below before you rely on that.

`--device cuda` refuses at load, by design, when the weights cannot be staged
into device memory (issue
[#1123](https://github.com/mudler/vllm.cpp/issues/1123)). The message names the
byte counts on both sides. That refusal is now **conditional on the lane**
(`ENG-EXPERT-STREAM-DEVICE` W0d, issue
[#1124](https://github.com/mudler/vllm.cpp/issues/1124)): with expert streaming
on, on a device whose kernels can dereference host memory, **and on a model
family that actually streams its experts**, the routed-expert towers are not
staged at all — their slices are read from the host slot store in place — so
what has to fit is the NON-expert remainder plus the slot arena rather than the
whole file.

The lane alone was not enough to produce a token. With it on, the checkpoint
loaded on `--device cuda` and then exhausted the machine inside its first
forward — zero decode steps, seven attempts, every one identical (issue
[#1299](https://github.com/mudler/vllm.cpp/issues/1299)) — because the DENSE
weights were resident twice: once as the host buffer and once as the device
staging copy, which on a part where device memory IS host memory comes out of the
same RAM. `VT_QWEN35_ALIAS_HOST_WEIGHTS` (default **on**, `docs/ENVIRONMENT.md`)
removes the second copy by handing the kernels the host bytes directly, and it is
what makes the CUDA arm decode at all. Set it to `0` for the same-binary A/B back
to the staging behaviour.

**It now decodes: 32/32 steps, at peak RSS 97.75 GiB of a 119.631 GiB box.**
Six limits, stated plainly rather than left to be discovered.

* **The device has to be probed capable, and most are not.** The condition is
  `cudaDevAttrPageableMemoryAccess AND cudaDevAttrIntegrated` — an integrated,
  unified part. A DISCRETE card answers false, keeps staging every tower and
  keeps the refusal. That is deliberate: a slot store the card cannot read is
  not a lane, and giving it one is later work on the same row.
* **The model has to be one of the families whose forward reads experts through
  the slot seam.** Today that is the Qwen3.5 MoE family:
  `Qwen3_5MoeForConditionalGeneration` and `Qwen3_5MoeForCausalLM`, which share
  one factory and one MoE block, and which a `qwen35moe` GGUF resolves to.
  `Qwen3MoeForCausalLM` (Qwen3-Coder) declares the same capability truthfully
  and NO GGUF load can reach it, because no `general.architecture` maps onto it
  — that gap is listed under `## Owed` in the row's spec. Every other
  architecture keeps the whole bound and keeps the refusal even with
  `VT_MOE_EXPERT_STREAM=1` set. `DeepseekV4ForCausalLM` is the case to have in
  mind: a `deepseek4` GGUF loads, its export carries the same `_exps.weight`
  tensor names, and its forward stages every one of those towers, so charging
  the device for a slot arena instead would under-count what the load really
  needs and turn a correct refusal into an out-of-memory first forward.
  `LagunaForCausalLM` is NOT that case and is not evidence for anything here: no
  `laguna` GGUF architecture arm exists, so a Laguna GGUF is refused as an
  unsupported architecture well before this check runs.
* **The checkpoint's expert towers have to KEEP the form the file stores them
  in, which means keep-quant OR keep-f16.** Those are the two residencies that
  read experts a slice at a time, and they are one arm rather than two: the
  loader sends both into the same stacked tower (`LoadExpertsStackedKq`), and the
  slice seam sizes a row with `vt::RowSizeBytes` and so never looks at the dtype.
  An F16 expert tower therefore gets the lane, and an operator holding one should
  not read this section and predict a refusal. The fp4-resident and the
  expand-to-bf16 arms of the same loader stage every tower like any other weight.
  So `VT_GGUF_KEEP_QUANT=0` — which turns keep-f16 off with it, because keep-f16
  rides the same condition — and an NVFP4 GGUF both keep the whole bound and keep
  the refusal on a device and a model that otherwise qualify. This is checked per
  file, against the residency this process resolved, and a file that mixes a kept
  tower with a staged one keeps the whole bound as well (issue
  [#1378](https://github.com/mudler/vllm.cpp/issues/1378)).
* **The correctness gate does NOT pass.** The 32 ids match the CPU arm for six
  tokens and diverge at the seventh. Both continuations are coherent, and the
  margins around it are measured and small: at that step the CPU arm's own
  second-ranked token is exactly the one the CUDA arm emitted, behind by 1.4% of
  the winning logit, and one step later the margin is 0.1%. **What CAUSES the
  divergence is NOT identified.** The host-weight alias is EXCLUDED, measured ON
  GB10 — same shapes, same algorithm, bit-identical output from a `cudaMalloc`
  operand and from a 256-aligned host one — but excluding one cause is not
  identifying another, and that the two arms simply run different GEMM kernels
  over a near-tie is a standing hypothesis rather than a reading.
  **2026-08-21: a router dump moved the failure UPSTREAM of the sampler.** On a
  later tree, source `cffe59b`, the two arms already select different MoE
  experts in the FIRST block of the FIRST forward, eight tokens before any
  emitted token differs, and they differ there in the router GEMM input rather
  than in anything the router does with it. One more cause is excluded by
  measurement: the router gate weights, whose fingerprint is identical on both
  arms. The two top-k implementations were checked as well and agreed with a
  plain lowest-index-wins rank of each arm's own logits, but only on 5 of the
  552 token-rows the dumps hold, so read that as a sample and not as a property
  of either implementation. A third probe then dumped the
  EMBEDDING OUTPUT, the hidden state before any GEMM, norm or attention touches
  it, and the two arms are BIT-IDENTICAL there: 0 of 40,960 bf16 values differ.
  So the weights are the same at both ends of the stack and the divergence
  starts in the compute inside the first block. The cause is STILL not
  identified, because the expert projections, the attention weights and the
  norms were never fingerprinted. The CUDA continuation also degenerates into a
  mechanical recursion after the tokens the two arms share, which a coin flip
  between two equally good tokens does not produce. Treat the CUDA arm as
  unverified against the CPU arm until that gate is settled, and **use
  `--device cpu` for this checkpoint today**: it is the arm every published
  number here was measured on.
* **No speed claim is attached.** `docs/BENCHMARKS.md` carries G0-SPEED as
  `VOID`, because a speed number behind a failing correctness gate is not a
  result. The CPU arm serves this checkpoint at a steady **11.05 s/token at 4000
  slots**, which is the count both recipes in this section set and the only count
  that figure holds for. Device access to host-resident weights on that part also has
  a recorded penalty, and this lane reads ~6.95 GB of expert bytes per token that
  way, so a CUDA arm slower than the CPU arm remains a real possible outcome.
  No published figure bounds this either way.
* **More slots is not a free knob, and the reason is the page cache rather than
  the arena.** The same binary at 8000 slots measured a 39.98-45.40 s/token
  median over two runs, and the second consumed all 30,625 MiB of the box's swap:
  the extra 9.27 GiB of arena takes the free memory the borrowed 370 GiB expert
  mapping is served out of. The arena is also measurably not what exhausted the
  box in [#1299](https://github.com/mudler/vllm.cpp/issues/1299) — a 64-slot
  0.15 GiB arena failed exactly where an 8000-slot 18.55 GiB one did — so this
  knob was never the lever there either.


## The same thing as config, and which one wins

The residency knobs are also config keys, under the `vllm_cpp` key of
`--offload-config` — the flag that already carries vLLM's weight-offload
document. `vllm-cli` takes the same flag, so the two recipes here differ only in
which binary they start, not in what each one can express. One flag covers both tiers: vLLM's own `uva`/`prefetch` keys move
weights from the device to host RAM, and the `vllm_cpp` key governs the tier
below that, where weights stay borrowed out of the file mapping.

```sh
./build/examples/vllm-server --model /models/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf \
  --offload-config '{"vllm_cpp":{"mmap":{"enabled":true,"prefault":false},
                                 "expert_stream":{"enabled":true,"slots":4000}}}'
```

| Key | Environment equivalent | Default |
|---|---|---|
| `vllm_cpp.mmap.enabled` | `VT_GGUF_MMAP` | on when weights stay quantized |
| `vllm_cpp.mmap.prefault` | `VT_GGUF_PREFAULT` | on with mmap residency — **set it `false` for a model larger than memory** |
| `vllm_cpp.expert_stream.enabled` | `VT_MOE_EXPERT_STREAM` | off |
| `vllm_cpp.expert_stream.slots` | `VT_MOE_EXPERT_STREAM_SLOTS` | `64`; a real model wants thousands |
| `vllm_cpp.expert_stream.slot_bytes` | `VT_MOE_EXPERT_STREAM_SLOT_BYTES` | the largest gate/up/down slice of the first MoE layer reached |
| `vllm_cpp.device_fit.weight_budget_bytes` | `VT_DEVICE_WEIGHT_BUDGET_BYTES` | the device's own probe (`cudaMemGetInfo` total on CUDA; no check elsewhere). `0` suppresses the load-time device-fit refusal; it is the only key here that accepts `0`, and a negative value is refused |

Every field is optional, and an absent field means unchanged, so an
`--offload-config` without a `vllm_cpp` key behaves exactly as it did before this
surface existed — with one difference, described below: a misspelled key is now an
error rather than being ignored. The same C ABI field carries it:
`vllm_model_params.offload_config` is one string holding both halves, so a library
client needs no new field.

**A second engine in one process is legal.** "Absent means unchanged" applies to the
install as well as to the parse: a later document is merged field by field over the
installed one, so `{"vllm_cpp":{"mmap":{"enabled":true}}}` on a second engine changes
`mmap` and leaves the first engine's `expert_stream` and slot count alone. Only two
things cannot be changed once a model has used them — whether expert streaming is on,
which is cached the first time it is asked, and the slot store's `slots x slot_bytes`
reservation, which is fixed when the store is built. A document that would change
either is refused at startup, naming the field and the value in force; a document that
omits it, or asks for exactly what is in force, is accepted.

**Precedence is `environment variable > config > built-in default`**, and it is
deliberate: the `VT_*` variables exist so a benchmark arm is switchable without
restarting the server with a new document, so `VT_MOE_EXPERT_STREAM=0` beats a
config `"enabled": true`. The engine prints one line at startup naming the fields
of the document it installed, and a second naming every variable that would win
over one of them, because a configuration silently overridden by something
exported weeks ago is the one way this precedence hurts. The first line reports
what was ASKED FOR, not what the engine resolves: the streaming answer is cached the
first time it is asked, so resolving it at startup would move that decision ahead of
the weight load. That constraint binds `expert_stream` alone — `prefault` and `slots`
could be resolved at startup, and `mmap` and `slot_bytes` need a built-in default only
their caller knows — and the line reports the document for all five so it reports one
kind of thing rather than a mixture. Read the two lines together: `expert_stream=on`
beside `VT_MOE_EXPERT_STREAM (expert_stream) OVERRIDES` means the document said on and
the variable decides.

**Where the config form reaches, and where it does not.** It reaches
`vllm-server`'s generate/chat path, `vllm-server`'s pooling/embedding path,
`vllm-cli`, and the C ABI's `vllm_model_params.offload_config`, which is the whole
of the library surface. All four take BOTH halves of the document, and the server
parses it once, before it reads the model's architecture, so a typo is refused at
startup whichever path the model then takes.

It does NOT reach the server's **transcription-only** path, and that path
**refuses the flag** rather than accepting it and doing nothing:

```text
server: fatal: --offload-config is not supported on a transcription-only model
(ParakeetForCTC). THE MISSING PART: this path serves /v1/audio/transcriptions
through ParakeetTranscriber, which loads its own weights and never builds an
engine, so neither vLLM's uva/prefetch weight offload nor vllm.cpp's vllm_cpp
weight-residency tier has a call site on it. ...
```

Use the environment form above on that path, or serve a text-generation or
embedding model. Recorded under `## Owed` in
[`.agents/specs/weight-residency-config.md`](../../.agents/specs/weight-residency-config.md)
with [#1195](https://github.com/mudler/vllm.cpp/issues/1195).
[#1135](https://github.com/mudler/vllm.cpp/issues/1135) is the issue this section
answered for the other three.

**A misspelled key is refused at startup, not ignored — at every level of the
document.** vLLM's own parser ignores a key it does not recognise, which is what
lets this extension share the flag, and it is also what would make
`{"vllm_cpp":{"mmapp":…}}` or `{"vllm-cpp":{…}}` start a server that quietly does
not borrow its weights, discovered later as an out-of-memory kill. The hyphenated
spelling is the likeliest typo of all, because every flag around it is hyphenated.
So the whole document is enumerated and the offender is named:

```text
offload config: unknown key "vllm_cpp.mmapp" (expected one of: mmap expert_stream device_fit)
offload config: unknown key "vllm-cpp" (expected one of: offload_backend uva prefetch vllm_cpp)
offload config: unknown key "uva.cpu_offload_GB" (expected one of: cpu_offload_gb cpu_offload_params)
```

Every level means every level, the mirrored sub-objects included. The enumeration once
stopped at the top level and inside `vllm_cpp`, which left the same hole one step down:
`{"uva":{"cpu_offload_GB":10}}` started a server with a 0 GiB offload budget the
operator believed was set.

The four legal top-level keys are `offload_backend`, `uva`, `prefetch` and
`vllm_cpp` — vLLM's three plus this extension — so a typo in the mirrored half
(`uvaa`, or `cpu_offload_gbb` inside it) is refused on the same terms. Refusing is what upstream does with its own
JSON config flags: vLLM builds its config dataclasses with a decorator that sets
`ConfigDict(extra="forbid")` (`vllm/config/utils.py:68-69`), which is why
`--kv-transfer-config` refuses an unknown key — and upstream has no
`--offload-config` at all, so no upstream-legal document is refused by this.

`VT_MOE_EXPERT_STREAM_STATS_EVERY` is **not** a config key, by decision: it
changes only how often the statistics line below is printed, so it is the
instrument rather than the configuration, and the config surface refuses it as an
unknown key rather than accepting and dropping it.

It applies to CPU keep-quant expert towers. On a device platform the expert
slice is already device-resident and is served unchanged, and turning streaming
on also disables the default-on grouped-MoE path, which stages the whole tower
and therefore cannot stream. The engine says that once on stderr rather than
silently doing no streaming.

**Read the statistics line before you believe any number you measure with it.**
The engine prints one every `VT_MOE_EXPERT_STREAM_STATS_EVERY` steps (default
16, `0` silences the periodic line), and **exactly one more when the process
ends**, whatever the run did:

```text
[expert-stream] steps=64 hits=141230 misses=37312 evictions=29312 fills=37312 bytes=92876505088 exhausted=0 advised=37312
```

**The final line is the one to read**, because it is the only one you are
guaranteed to get. The periodic line is skipped whenever the step count is not a
multiple of the interval, so a healthy five-token run prints none of them at the
default 16; and it used to be skipped on `steps == 0` as well, which meant the
one run that most needed reporting — the one where the step boundary is never
reached — printed nothing at all. Treating absence as failure therefore reported
VOID on a working lane. The final line crosses both of those skips, so it is
printed even on a run of zero steps.

Two of the fields decide whether the run is measuring anything at all:

- `steps` must advance. If the final line says `steps=0` the decode step
  boundary is not being reached, and the cache stops serving as soon as it
  fills — it will fall back to the memory mapping for the rest of the run.
- `exhausted` must stay 0. Anything above 0 means slices were refused and read
  from the memory mapping instead, which is the slow path streaming exists to
  replace. The usual cause is a budget smaller than one step's working set:
  raise `VT_MOE_EXPERT_STREAM_SLOTS`.

Read it together with the `[expert-stream] ON slots=...` banner, which is printed
once when the lane builds its store. The four shapes are:

| Banner | Final line | What happened |
|---|---|---|
| absent | absent | Nothing reached the streamed seam. A CUDA run (a device-resident expert is served unchanged), a checkpoint whose experts are not keep-quant towers, or a prompt that never reached an MoE layer |
| present | present | The lane ran. Read `steps` and `exhausted` |
| present | absent, and nothing called `ExpertStreamFlushStats` | The process did not reach its static destructors: a crash, a signal, or `_exit` |
| present | absent, because `ExpertStreamFlushStats` was called | The internal gate seam took the process's single print, so teardown had none left to make. No shipped command or server path calls it, so an operator never reaches this shape |

The last two shapes are keyed on the CALL and not on what stderr looks like,
because stderr cannot separate them. `ExpertStreamFlushStats` prints the same
line in the same shape as the periodic report, so "a statistics line already
appeared mid-run" is also what a healthy run of 16 steps that then crashes
produces. What distinguishes the two is whether the seam was called, and only a
gate calls it.

A run whose `steps` is 0, or whose `exhausted` is large, is not a measurement of
streaming, whatever the startup line said. See
[`docs/ENVIRONMENT.md`](../ENVIRONMENT.md) for every knob and its parsing rules.


## `--device cuda` refuses a checkpoint it cannot hold

Streaming is a **host** capability. The GGUF mapping is borrowed in place on the
CPU path, so a routed-expert tower costs no resident bytes, which is the whole
reason a 369.96 GiB checkpoint serves on a 119.631 GiB box. A weight-staging
device has no such lane: it copies every tower into device memory, one
`cudaMalloc` per stacked `[E*N,K]` tower.

For `Qwen3.8-2.4T-A95B UD-Q1_0` that is 276 towers of 1,275,068,416 bytes plus
three of 2,818,572,288, so 335.62 GiB in total, against a pool `cudaMemGetInfo`
reports as
128,452,956,160 bytes (119.631 GiB). Until that lane exists
([#1124](https://github.com/mudler/vllm.cpp/issues/1124)), the engine **refuses
at load** and names what is missing:

```text
device 'cuda' cannot serve this GGUF: staging its weights needs at least N bytes
(X GiB) of device memory across T tensors, the largest single allocation being M
bytes (Y GiB, '<tensor>'), and this device's memory pool is B bytes (Z GiB).
THE MISSING PART: ... there is no device-side expert slot store and no device
streaming lane ... Use device=cpu, which serves this checkpoint today, or a
checkpoint that fits the pool.
```

It used to load for 26 minutes, report ready, and then die on the first request
with `vt cuda: cudaMalloc: out of memory` from inside the engine's busy loop
([#1123](https://github.com/mudler/vllm.cpp/issues/1123)).

The refusal is keyed on the measured condition and not on the device or the file
format, so **a GGUF that fits the pool still loads on `--device cuda`**. Three
things it deliberately does not do:

- it never fires on a platform that does not stage weights, so every
  `--device cpu` load is unchanged;
- it never fires when no budget is known. Today exactly one platform stages
  weights (CUDA) and exactly one probes a budget (CUDA, with `cudaMemGetInfo`),
  so **every NVIDIA GPU this build runs on — discrete or GB10 — gets both the
  probe and the refusal**, while ROCm, Vulkan and Metal answer
  `needs_weight_staging() == false`: they read the GGUF mapping where it already
  lies, so there is no staging allocation to fail and nothing for this check to
  decide. What is owed there is the `Backend::DeviceMemoryInfo` probe CUDA does
  not implement ([#1126](https://github.com/mudler/vllm.cpp/issues/1126)), which
  is a different capability;
- it counts **weights only**. The KV cache, activations, scratch pools and the
  driver context are not in the bound, so a checkpoint just under the pool
  passes this check and can still fail later;
- it can also count a little **too much**: a tensor present in the file that this
  load will not stage — the MTP / `nextn` block on a load with no speculator, 8.33
  GiB of the measured 369.96 GiB checkpoint — is still in the sum, so a budget in
  that narrow window refuses a weight set that would have fitted. Raise
  the budget if you land in it
  ([#1136](https://github.com/mudler/vllm.cpp/issues/1136)).

**Moving the budget.** Lower it when something else lives in the pool, or raise
it (or set `0`) to suppress the refusal and get the late failure back. It does
not make the model fit. Two ways to say it, and the first beats the second:

```sh
VT_DEVICE_WEIGHT_BUDGET_BYTES=68719476736 ./build/examples/vllm-server --model ...
./build/examples/vllm-server --model ... \
  --offload-config '{"vllm_cpp":{"device_fit":{"weight_budget_bytes":68719476736}}}'
```

The config key is the same `--offload-config` document the residency knobs use,
so one flag still covers weight placement
([#1127](https://github.com/mudler/vllm.cpp/issues/1127)). `0` from either input
suppresses the refusal. The environment variable takes decimal digits only: a
value with a sign, a space or trailing garbage is ignored and falls through to
the config, then to the probe, because reading a typo as `0` would silently
disable the guard. A malformed config value cannot get that far, because the
parser refuses it at startup.

**The instrument matters here.** `nvidia-smi
--query-gpu=memory.total,memory.free,memory.used` answers `[N/A], [N/A], [N/A]`
on a GB10, because host and device share one pool. `cudaMemGetInfo` answers
honestly, and its `total` is EXACTLY `/proc/meminfo MemTotal`
(125442340 kB) times 1024. Do not size this from `nvidia-smi`.


## Qwen3.8-2.4T-A95B `UD-Q1_0`: 370 GiB served from a 119 GiB box

A 2.4-trillion-parameter mixture-of-experts checkpoint, three times the size of
the machine's memory, loads and answers on one DGX Spark. This section is the
recipe. The mechanism it drives is the previous section,
[expert streaming guide](../guides/expert-streaming.md),
which owns the config schema, the precedence rule, the statistics line, the slot
count warning and what each device can serve. This section links them rather than
restating them. It repeats three of their facts on purpose: which device to use,
the expert bytes a token reads, and the two streaming decode figures in
[What decode costs](#what-decode-costs-and-why-the-ceiling-is-where-it-is). A
recipe that leaves those out is not a recipe. Each of the three has one record,
so a correction has to change both places. The decode figures are
`ENG-EXPERT-STREAM-DEVICE` W0e in
[`.agents/benchmark-record.md`](../../.agents/benchmark-record.md).

**Read the speed before you spend the download.** Steady decode on the recipe
below is measured in seconds per token, and the floor under it is storage rather
than this implementation. This is a capacity result, not an interactive one.
[What decode costs](#what-decode-costs-and-why-the-ceiling-is-where-it-is) gives
the figure and the arithmetic behind it.

**Use `--device cpu` for this checkpoint.** `--device cuda` loads and decodes it
too, and its token gate against the CPU arm does not pass, so every number below
was measured on the CPU arm. The previous section states that arm's six limits.

## The exact weights

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

**Ten shards, and the count is part of every file name.** A GGUF split writes
the total into each member's name, so `-of-00008` and `-of-00010` name different
files, and the wrong one gives a file-not-found after a 370 GiB download. The
two recipes in the previous section carried `-of-00008` until this change
([#1420](https://github.com/mudler/vllm.cpp/issues/1420)). The count is settled
against the artifact and not against a document: shard 1's own metadata declares
`split.count = 10` and `split.tensors.count = 1702`, its sha256 recomputed from
the mirrored copy equals the value above, and the ten files sum to exactly the
byte total above. Shard 1 holds **no tensors at all**. It is the metadata and the
split declaration, so it is the file that says what the other nine are.

**A repo id alone is not a pin**, because a quantized checkpoint gets
re-quantized in place under an unchanged name. Shard 1's digest was recomputed
from the mirrored copy. Shard 2's is the download manifest's, and its byte count
was recomputed.

```sh
hf download unsloth/Qwen3.8-2.4T-A95B-GGUF \
  --revision 567d3e6ac26c5474b18311e619c04350fb9a5556 \
  --include "UD-Q1_0/*" \
  --local-dir ./qwen3.8-2.4t-a95b-gguf
```

The files land under a `UD-Q1_0/` subdirectory of `--local-dir`, because that is
where they live in the repo. Point `--model` at a copy on **local NVMe**. A
network filesystem puts an uncontrolled variable in front of the expert reads
that every token makes.

**The encoding has no upstream reference.** `UD-Q1_0` stores its expert towers as
`IQ1_XXXS`, which upstream llama.cpp does not define. The encoding exists only in
the `unslothai/llama.cpp` fork, pinned as a secondary oracle in
[`.agents/oracles/llama-cpp-unsloth.md`](../../.agents/oracles/llama-cpp-unsloth.md).
That fork is recorded `gateable = no`, because it has not been shown to build and
run this model, and [#933](https://github.com/mudler/vllm.cpp/issues/933) owes
the measurement. There is therefore no token-exact denominator for anything below.

## Build and serve

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

Five things in that command are load-bearing.

- **`--model` takes shard 1, not the directory.** A directory sends the loader
  down the HuggingFace branch, which fatals on a missing `config.json` before it
  looks for a GGUF. Given shard 1 the reader finds its nine siblings from the
  `-NNNNN-of-MMMMM.gguf` naming and cross-checks `split.count`.
- **`prefault: false` is the setting that decides whether this works.**
  Pre-faulting is **on** by default, and it is the right default for a model that
  fits: it walks every borrowed span at load, so the first-touch faults do not
  land inside the timed prefill. For 335.62 GiB of expert towers that cannot fit,
  it reads the whole checkpoint to populate a page cache that cannot hold it.
- **`mmap: true` confirms the default rather than enabling it.** It is already on
  wherever the weights stay quantized, and it is what makes the checkpoint fit at
  all: an expert tower is borrowed from the file mapping and costs zero anonymous
  bytes, so only the dense remainder becomes resident.
- **`expert_stream` is off by default, and this recipe turns it on at 4000
  slots.** That count is the one the published decode figure was measured at, and
  the previous section explains why 8000 is worse rather than better.
- **`--device cpu`.** The note at the top of this section says why.

`--max-num-seqs 1` and a small `--max-model-len` keep the KV cache out of the
way. Nothing is batched at this speed, and the capacity argument itself holds
only at low concurrency: at high concurrency every step touches most of the
experts and the working set stops being one.

The recorded runs set the equivalent environment variables rather than the
config document: `VT_GGUF_PREFAULT=0`, `VT_MOE_EXPERT_STREAM=1` and
`VT_MOE_EXPERT_STREAM_SLOTS=4000`. The two forms are the same switches, and a
variable beats a config field wherever both are set.

They also ran a different binary. Every W0e and W0f figure below comes from
`benchmarks/expert_stream_device_w0e.cpp`, a purpose-built C ABI client that
reports the token ids, a per-step timestamp and the expert-stream counters
together, which no shipped command does. The 16 August 2026 run is the one
exception: it served through `vllm-server`, as the command above does. At
seconds per token, the server's HTTP and SSE framing sits far below the
run-to-run spread recorded below.

## What the load costs

Expect to wait. Two runs of this arm are recorded on `dgx:gpu0`, a GB10 with
119.631 GiB of unified memory reading the checkpoint from local NVMe, with the
page cache dropped before each one (`ENG-EXPERT-STREAM-DEVICE` W0e, 18 and
19 August 2026, [`.agents/benchmark-record.md`](../../.agents/benchmark-record.md)):

| Axis | Run 1 | Run 2 |
|---|---|---|
| load | 271.1 s | 255.7 s |
| first token | 85.90 s | 79.09 s |
| peak resident set | 86.5 GiB | 86.5 GiB |
| peak swap | not sampled | 6 883 MiB |

Resident memory after the load settles at about **62 GiB** of 119 GiB, measured
at 62.45 GiB on the same-lease CPU control run of `ENG-EXPERT-STREAM-DEVICE`
W0f. That is the
dense remainder plus the KV cache and the runtime, and it agrees with what the
checkpoint's own tensor table predicts: 21.56 GiB of `attn_qkv` and 17.25 GiB of
`ssm_out` expanded to bf16, plus 5.81 GiB of embeddings and F32 norms, so
44.6 GiB before the KV cache and the runtime. The other 335.62 GiB is mapped,
not copied. **The model does not fit because of streaming. It fits because of
borrowing.**

Check readiness against the model list rather than against the process:

```sh
curl -sf http://127.0.0.1:8899/v1/models
```

```sh
curl -s http://127.0.0.1:8899/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf",
       "prompt":"Q: What is the capital of France? A:","max_tokens":4}'
```

The 16 August 2026 run, which served with streaming off, answered
` Paris. Q: What`. That is the whole point: the output is coherent, so the
one-bit encoding and the borrowed-tower path are both faithful enough to serve.
The four W0e runs drive a fixed prompt of token ids instead of this request, and
all four returned the same 32 ids, which detokenize to ` Paris. Paris is a city
located in the northern part of France, on the Seine River. It is the largest
city in France and is known for its iconic`.

## What decode costs, and why the ceiling is where it is

Every figure here comes from the box named above.

| Arm | Steady decode | Where it comes from |
|---|---|---|
| streaming on, 4000 slots | **11.05 s/token** | W0e rep 2, median over steps 4 to 32 |
| streaming on, 8000 slots | 39.98 and 45.40 s/token | W0e, the medians of two reps |
| streaming off | 66.7 s/token | 16 August 2026, streaming not yet enabled |

That 4000-slot figure has a min of 9.43 and a max of 13.25 over its window, and
rep 1 of the same arm gives 11.22, which is 1.54% above it. **The
streaming-off row carries no ratio against the other two**, because it was taken
on a different source tree on a different date. The two slot counts came from one
binary on one lease and are comparable with each other; the previous section
carries that comparison.

**A bigger cache came out slower**, which is why this recipe sets 4000 slots.
The previous section states the reason and its evidence.

Do not quote a first-token time as a decode number. Token 1 carries the prefill
and the cold expert set. From token 2 onward you are watching steady state. The
complete measurement record is [docs/BENCHMARKS.md](../BENCHMARKS.md).

The arithmetic behind those seconds is short, and it decides everything. The
first three rows are read from the checkpoint's own metadata:

| Quantity | Value |
|---|---|
| blocks (`qwen35moe.block_count`) | 93 |
| experts routed per block, of `qwen35moe.expert_count` | 10 of 512 |
| projections per routed expert | 3 |
| expert slices per token | 2790 |
| bytes per slice | 2 490 368, that is 2.375 MiB |
| expert working set per token | **6.95 GB**, that is 6.47 GiB |
| slots this recipe reserves | 4000, a 9.28 GiB arena |

That figure is a working set and not an I/O rate, because the slot cache serves
part of it from memory. The recorded 32-token run at 4000 slots counted 37 096
hits against 58 538 misses.

**The floor is storage, not software.** 6.95 GB at the roughly 5 GB/s an NVMe of
this class sustains is 1.39 s/token whatever the code does, which is 0.72 tok/s.
Reaching 3 tok/s would demand about 21 GB/s of expert bandwidth, so most of those
reads would have to come from memory instead. The arena holds 4000 slices against
the 2790 a token needs, under one and a half tokens of working set, and
top-10-of-512 routing does not give consecutive tokens enough reuse to close the
rest. **If you need conversational speed from this model you need more memory or
fewer active parameters, not better software.**

## What this does not establish

- **The quantization is extreme.** The expert towers hold 1.1875 bits per weight,
  and they are about 97% of the parameters. The output is coherent; this is not
  the configuration to judge the model's quality by.
- **There is no oracle.** No entry in the oracle table runs this checkpoint on
  this hardware, so there is no token-exact and no throughput denominator. Every
  figure above is an absolute measurement of this implementation, compared
  against nothing.
- **One request at a time.** Nothing here says anything about concurrency, and
  the capacity argument stops holding as concurrency rises.
- **One box.** Every number was taken on one DGX Spark GB10 with the checkpoint
  on local NVMe. Different storage or a different host changes them.
- **Nothing here is a `--device cuda` number.** That arm decodes this checkpoint
  and its token gate against the CPU arm fails. The previous section's sixth
  limit states what follows for its speed axis.
