# Stream routed experts from disk

Use expert streaming when the checkpoint is larger than available memory.

A mixture-of-experts checkpoint larger than the box can hold can be run by
keeping the routed-expert weights on disk and paging slices into a bounded
resident cache. It is **off by default** and it is a **capacity** feature, not a
throughput one: it targets single-user and low-concurrency use, and at high
concurrency every step touches most of the experts, so there is nothing left to
save.

```sh
VT_MOE_EXPERT_STREAM=1 \
VT_MOE_EXPERT_STREAM_SLOTS=4000 \
  ./build/examples/vllm-cli --model /models/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf \
                   --prompt "The capital of France is" --max-tokens 16
```

[Qwen3.8 2.4T](../models/qwen3-8-2-4t.md) is the worked example, with measured
load, decode, and memory figures for a 369.97 GiB checkpoint on a 119.631 GiB
box.

## Borrowing comes first, streaming comes second

Two separate mechanisms let a checkpoint exceed memory, and confusing them leads
to tuning the wrong one.

**Borrowing** is what makes the checkpoint fit. The GGUF mapping is read in place
on the CPU path, so a routed-expert tower costs no resident bytes and only the
dense remainder is copied. Borrowing needs `mmap` on and quantized residency, and
it is already the default wherever the weights stay quantized.

**Streaming** is what makes the reads cheaper. It keeps a fixed arena of recently
used expert slices in memory, so part of each token's working set is served from
RAM instead of from storage. Streaming is off by default.

A model fits because of borrowing. Turning streaming on without borrowing does
not make a checkpoint fit.

## Configure it

The knobs are environment variables and also config keys, under the `vllm_cpp`
key of `--offload-config`. One flag covers both tiers: vLLM's own `uva` and
`prefetch` keys move weights from the device to host RAM, and the `vllm_cpp` key
governs the tier below that, where weights stay borrowed out of the file mapping.

```sh
./build/examples/vllm-server --model /models/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf \
  --offload-config '{"vllm_cpp":{"mmap":{"enabled":true,"prefault":false},
                                 "expert_stream":{"enabled":true,"slots":4000}}}'
```

| Key | Environment equivalent | Default |
|---|---|---|
| `vllm_cpp.mmap.enabled` | `VT_GGUF_MMAP` | on when weights stay quantized |
| `vllm_cpp.mmap.prefault` | `VT_GGUF_PREFAULT` | on with mmap residency. **Set it `false` for a model larger than memory** |
| `vllm_cpp.expert_stream.enabled` | `VT_MOE_EXPERT_STREAM` | off |
| `vllm_cpp.expert_stream.slots` | `VT_MOE_EXPERT_STREAM_SLOTS` | `64`. A real model wants thousands |
| `vllm_cpp.expert_stream.slot_bytes` | `VT_MOE_EXPERT_STREAM_SLOT_BYTES` | the largest gate/up/down slice of the first MoE layer reached |
| `vllm_cpp.device_fit.weight_budget_bytes` | `VT_DEVICE_WEIGHT_BUDGET_BYTES` | the device's own probe, `cudaMemGetInfo` total on CUDA and no check elsewhere. `0` suppresses the load-time device-fit refusal. It is the only key here that accepts `0`, and a negative value is refused |

`VT_MOE_EXPERT_STREAM_STATS_EVERY` is **not** a config key, by decision. It
changes only how often the statistics line is printed, so it is the instrument
rather than the configuration, and the config surface refuses it as an unknown
key rather than accepting and dropping it.

### Which input wins

**Precedence is environment variable, then config, then built-in default.** The
`VT_*` variables exist so a benchmark arm is switchable without restarting the
server with a new document, so `VT_MOE_EXPERT_STREAM=0` beats a config
`"enabled": true`.

The engine prints one line at startup naming the fields of the document it
installed, and a second naming every variable that would win over one of them. A
configuration silently overridden by something exported weeks ago is the one way
this precedence hurts. Read the two lines together: `expert_stream=on` beside
`VT_MOE_EXPERT_STREAM (expert_stream) OVERRIDES` means the document said on and
the variable decides.

The first line reports what was **asked for**, not what the engine resolves. The
streaming answer is cached the first time it is asked, so resolving it at startup
would move that decision ahead of the weight load.

### Where the config form reaches

It reaches `vllm-server`'s generate and chat path, `vllm-server`'s pooling and
embedding path, `vllm-cli`, and the C ABI's `vllm_model_params.offload_config`,
which is the whole of the library surface. All four take both halves of the
document, and the server parses it once, before it reads the model's
architecture, so a typo is refused at startup whichever path the model then
takes.

It does **not** reach the server's transcription-only path, and that path
**refuses the flag** rather than accepting it and doing nothing. That path serves
`/v1/audio/transcriptions` through `ParakeetTranscriber`, which loads its own
weights and never builds an engine, so neither weight-offload tier has a call
site on it. Use the environment form there, or serve a text-generation or
embedding model. Recorded under `## Owed` in
[`.agents/specs/weight-residency-config.md`](../../.agents/specs/weight-residency-config.md)
with [#1195](https://github.com/mudler/vllm.cpp/issues/1195).

**A second engine in one process is legal.** "Absent means unchanged" applies to
the install as well as to the parse, so a later document is merged field by field
over the installed one. Only two things cannot be changed after a model has used
them: whether expert streaming is on, which is cached the first time it is asked,
and the slot store's `slots x slot_bytes` reservation, which is fixed when the
store is built. A document that would change either is refused at startup, naming
the field and the value in force.

### A misspelled key is refused, not ignored

vLLM's own parser ignores a key it does not recognize, which is what lets this
extension share the flag. It is also what would make `{"vllm_cpp":{"mmapp":...}}`
or `{"vllm-cpp":{...}}` start a server that quietly does not borrow its weights,
discovered later as an out-of-memory kill. The hyphenated spelling is the
likeliest typo of all, because every flag around it is hyphenated. So the whole
document is enumerated and the offender is named:

```text
offload config: unknown key "vllm_cpp.mmapp" (expected one of: mmap expert_stream device_fit)
offload config: unknown key "vllm-cpp" (expected one of: offload_backend uva prefetch vllm_cpp)
offload config: unknown key "uva.cpu_offload_GB" (expected one of: cpu_offload_gb cpu_offload_params)
```

Every level means every level, the mirrored sub-objects included. The enumeration
once stopped at the top level and inside `vllm_cpp`, which left the same hole one
step down: `{"uva":{"cpu_offload_GB":10}}` started a server with a 0 GiB offload
budget the operator believed was set.

The four legal top-level keys are `offload_backend`, `uva`, `prefetch`, and
`vllm_cpp`, which is vLLM's three plus this extension. Refusing is what upstream
does with its own JSON config flags: vLLM builds its config dataclasses with a
decorator that sets `ConfigDict(extra="forbid")` (`vllm/config/utils.py:68-69`),
which is why `--kv-transfer-config` refuses an unknown key. Upstream has no
`--offload-config` at all, so no upstream-legal document is refused by this.

## Read the statistics line before you believe a number

The engine prints one line every `VT_MOE_EXPERT_STREAM_STATS_EVERY` steps
(default 16, `0` silences the periodic line), and **exactly one more when the
process ends**, whatever the run did:

```text
[expert-stream] steps=64 hits=141230 misses=37312 evictions=29312 fills=37312 bytes=92876505088 exhausted=0 advised=37312
```

**The final line is the one to read**, because it is the only one you are
guaranteed to get. The periodic line is skipped whenever the step count is not a
multiple of the interval, so a healthy five-token run prints none of them at the
default 16. It used to be skipped on `steps == 0` as well, which meant the one run
that most needed reporting, the one where the step boundary is never reached,
printed nothing at all. Treating absence as failure therefore reported VOID on a
working lane. The final line crosses both skips.

Two fields decide whether the run is measuring anything at all:

- `steps` must advance. If the final line says `steps=0`, the decode step
  boundary is not being reached, and the cache stops serving as soon as it fills.
  It falls back to the memory mapping for the rest of the run.
- `exhausted` must stay 0. Anything above 0 means slices were refused and read
  from the memory mapping instead, which is the slow path streaming exists to
  replace. The usual cause is a budget smaller than one step's working set. Raise
  `VT_MOE_EXPERT_STREAM_SLOTS`.

Read it together with the `[expert-stream] ON slots=...` banner, printed once
when the lane builds its store. The four shapes are:

| Banner | Final line | What happened |
|---|---|---|
| absent | absent | Nothing reached the streamed seam. A CUDA run, a checkpoint whose experts are not keep-quant towers, or a prompt that never reached an MoE layer |
| present | present | The lane ran. Read `steps` and `exhausted` |
| present | absent, and nothing called `ExpertStreamFlushStats` | The process did not reach its static destructors: a crash, a signal, or `_exit` |
| present | absent, because `ExpertStreamFlushStats` was called | The internal gate seam took the process's single print. No shipped command or server path calls it, so an operator never reaches this shape |

The last two shapes are keyed on the call and not on what stderr looks like,
because stderr cannot separate them. `ExpertStreamFlushStats` prints the same line
in the same shape as the periodic report, so "a statistics line already appeared
mid-run" is also what a healthy run of 16 steps that then crashes produces.

A run whose `steps` is 0, or whose `exhausted` is large, is not a measurement of
streaming, whatever the startup line said.

## What each device can serve

Streaming is a **host** capability, and it applies to CPU keep-quant expert
towers. On a device platform the expert slice is already device-resident and is
served unchanged. Turning streaming on also disables the default-on grouped-MoE
path, which stages the whole tower and therefore cannot stream. The engine says
that once on stderr rather than silently doing no streaming.

A weight-staging device copies every tower into device memory, one `cudaMalloc`
per stacked `[E*N,K]` tower, and **refuses at load** when the total exceeds the
pool. The refusal names the byte counts on both sides:

```text
device 'cuda' cannot serve this GGUF: staging its weights needs at least N bytes
(X GiB) of device memory across T tensors, the largest single allocation being M
bytes (Y GiB, '<tensor>'), and this device's memory pool is B bytes (Z GiB).
THE MISSING PART: ... there is no device-side expert slot store and no device
streaming lane ... Use device=cpu, which serves this checkpoint today, or a
checkpoint that fits the pool.
```

The refusal is keyed on the measured condition and not on the device or the file
format, so a GGUF that fits the pool still loads on `--device cuda`. Three things
it deliberately does not do:

- It never fires on a platform that does not stage weights, so every
  `--device cpu` load is unchanged.
- It never fires when no budget is known. Today exactly one platform stages
  weights and probes a budget, which is CUDA with `cudaMemGetInfo`, so every
  NVIDIA GPU this build runs on gets both the probe and the refusal. ROCm,
  Vulkan, and Metal answer `needs_weight_staging() == false`, because they read
  the GGUF mapping where it already lies. The `Backend::DeviceMemoryInfo` probe
  they lack is owed by
  [#1126](https://github.com/mudler/vllm.cpp/issues/1126).
- It counts **weights only**. The KV cache, activations, scratch pools, and the
  driver context are not in the bound, so a checkpoint just under the pool passes
  this check and can still fail later.

It can also count a little **too much**. A tensor present in the file that this
load will not stage, for example the MTP or `nextn` block on a load with no
speculator, is still in the sum, so a budget in that narrow window refuses a
weight set that would have fitted
([#1136](https://github.com/mudler/vllm.cpp/issues/1136)).

**Moving the budget.** Lower it when something else lives in the pool, or raise it
(or set `0`) to suppress the refusal and get the late failure back. It does not
make the model fit. Two ways to say it, and the first beats the second:

```sh
VT_DEVICE_WEIGHT_BUDGET_BYTES=68719476736 ./build/examples/vllm-server --model ...
./build/examples/vllm-server --model ... \
  --offload-config '{"vllm_cpp":{"device_fit":{"weight_budget_bytes":68719476736}}}'
```

The environment variable takes decimal digits only. A value with a sign, a space,
or trailing garbage is ignored and falls through to the config, then to the
probe, because reading a typo as `0` would silently disable the guard. A
malformed config value cannot get that far, because the parser refuses it at
startup.

**The instrument matters here.** `nvidia-smi
--query-gpu=memory.total,memory.free,memory.used` answers `[N/A], [N/A], [N/A]`
on a GB10, because host and device share one pool. `cudaMemGetInfo` answers
honestly, and its `total` is exactly `/proc/meminfo MemTotal` times 1024. Do not
size this from `nvidia-smi`.

### The device streaming lane

With expert streaming on, on a device whose kernels can dereference host memory,
and on a model family that streams its experts, the routed-expert towers are not
staged at all. Their slices are read from the host slot store in place, so what
has to fit is the non-expert remainder plus the slot arena rather than the whole
file ([#1124](https://github.com/mudler/vllm.cpp/issues/1124)).

Three conditions gate it, and all must hold at once:

- **The device is probed capable**, meaning
  `cudaDevAttrPageableMemoryAccess AND cudaDevAttrIntegrated`. A discrete card
  answers false and keeps the refusal, because a slot store the card cannot read
  is not a lane.
- **The model family reads experts through the slot seam.** Today that is the
  Qwen3.5 MoE family, `Qwen3_5MoeForConditionalGeneration` and
  `Qwen3_5MoeForCausalLM`, which a `qwen35moe` GGUF resolves to.
  `Qwen3MoeForCausalLM` declares the same capability truthfully and no GGUF load
  can reach it, because no `general.architecture` maps onto it.
  `DeepseekV4ForCausalLM` is the case to have in mind: a `deepseek4` GGUF loads
  and its export carries the same `_exps.weight` tensor names, but its forward
  stages every tower, so charging the device for a slot arena instead would
  under-count what the load needs. `LagunaForCausalLM` is not that case and is
  not evidence for anything here: no `laguna` GGUF architecture arm exists, so a
  Laguna GGUF is refused as an unsupported architecture well before this check
  runs.
- **The expert towers keep the form the file stores them in**, which means
  keep-quant or keep-f16. Those two are one arm rather than two: the loader sends
  both into the same stacked tower, and the slice seam sizes a row with
  `vt::RowSizeBytes` and so never looks at the dtype. The fp4-resident and the
  expand-to-bf16 arms stage every tower like any other weight, so
  `VT_GGUF_KEEP_QUANT=0` and an NVFP4 GGUF both keep the refusal. This is checked
  per file, and a file that mixes a kept tower with a staged one keeps the whole
  bound ([#1378](https://github.com/mudler/vllm.cpp/issues/1378)).

On a part where device memory is host memory, the dense weights would otherwise
be resident twice, once as the host buffer and once as the device staging copy.
`VT_QWEN35_ALIAS_HOST_WEIGHTS`, default **on**, removes the second copy by handing
the kernels the host bytes directly, and it is what makes that arm decode at all
([#1299](https://github.com/mudler/vllm.cpp/issues/1299)). Set it to `0` for the
same-binary A/B back to the staging behavior.

**The correctness gate on that arm does not pass.** See
[the CUDA arm](../models/qwen3-8-2-4t.md#the-cuda-arm-and-why-not-to-use-it) for
what diverges and what was excluded as a cause.

## Where the rest lives

| You want | Read |
|---|---|
| A worked example with measured figures | [Qwen3.8 2.4T](../models/qwen3-8-2-4t.md) |
| Every knob and its parsing rules | [`docs/ENVIRONMENT.md`](../ENVIRONMENT.md) |
| The device-to-host weight offload tier | [`docs/WEIGHT-OFFLOAD.md`](../WEIGHT-OFFLOAD.md) |
| Design rationale and evidence | [`.agents/specs/expert-streaming.md`](../../.agents/specs/expert-streaming.md) |
