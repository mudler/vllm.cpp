# Weight offload

Weight offload keeps model **weights** outside device memory so that a model
larger than the device can still load. It is a different subject from
[KV offload](KV-OFFLOAD.md), which keeps **KV blocks** outside the paged cache.
The two use separate flags and do not interact.

Selection mirrors vLLM's own configuration: `--offload-config '<json>'`, taking
the same JSON object vLLM's `OffloadConfig` takes, so a config written for vLLM
is accepted here.

> **Enabling weight offload fails startup today, on every model.** No loader
> consults the offloader yet, so the engine refuses the configuration rather
> than accept a budget that would free nothing. Read
> [What works today](#what-works-today) before you enable anything.

## What works today

| Part | State |
|---|---|
| `--offload-config` parsing and validation | Works. A malformed document, an unknown backend, or a validator violation is refused at startup |
| The `uva` backend object and its byte budget | Built. It answers the offload decision and counts approved bytes |
| The `prefetch` backend | Not built. A config that selects it is accepted and reported as unbuilt |
| A loader that asks the offloader and keeps a weight off the device | **Not wired.** This is the part that frees memory, and it does not exist yet |
| A model that accepts an enabled offload | **None.** Every architecture is refused at startup — see [What the engine refuses](#what-the-engine-refuses-and-what-it-only-warns-about) |
| Pinned host copies and device views | Not built |

So the honest summary is: you can write and validate a weight-offload
configuration, and you cannot yet run with one enabled. A budget that silently
frees nothing is worse than a budget the engine refuses, so the engine refuses
it.

A configuration that leaves offloading **disabled** is unaffected: the guard
only fires when `is_offloading_enabled()` is true, so parsing, validation, and
the resolved-backend report all still work for inspecting a config.

Progress is tracked in [issue #797](https://github.com/mudler/vllm.cpp/issues/797).

## The flag

```sh
vllm-server --model /path/to/model \
  --offload-config '{"offload_backend":"uva","uva":{"cpu_offload_gb":10}}'
```

The same document is available through the C API as the `offload_config` field
of `vllm_model_params` (ABI v21). `NULL` or an empty string means no offload,
which is the default engine.

### Fields

| Field | Default | Meaning |
|---|---|---|
| `offload_backend` | `auto` | `auto`, `uva`, or `prefetch`. Under `auto` the engine selects `prefetch` when `offload_group_size` is above 0, then `uva` when `cpu_offload_gb` is above 0, and otherwise offloads nothing |
| `uva.cpu_offload_gb` | `0` | The space in GiB to keep off the device, per GPU. `0` means no offload |
| `uva.cpu_offload_params` | empty | Parameter-name segments to target. Empty means every parameter is eligible until the budget is spent |
| `prefetch.offload_group_size` | `0` | Group every N layers and offload the last few of each group. `0` disables it |
| `prefetch.offload_num_in_group` | `1` | How many layers to offload in each group. Must not exceed `offload_group_size` |
| `prefetch.offload_prefetch_step` | `1` | How many layers to prefetch ahead. Must be 1 or more when prefetch is enabled |
| `prefetch.offload_params` | empty | Parameter-name segments to target. Empty means every parameter of each offloaded layer |

### Targeting by name segment

`cpu_offload_params` matches whole dotted segments, not substrings. For the
parameter `mlp.experts.w2_weight`:

| Segment | Matches | Reason |
|---|---|---|
| `experts` | Yes | A whole segment |
| `experts.w2_weight` | Yes | Two whole segments |
| `expert` | No | Not a whole segment |
| `w2` | No | Not a whole segment |
| `w2_weight` | Yes | The final segment |

The last row is the reason the rule exists: `w2_weight` must not also match
`mlp.experts.w2_weight_scale`, because a quantized weight and its scale are
different tensors with different sizes.

### The two empty-set defaults are not the same

An empty `uva.cpu_offload_params` means "offload without selecting, until the
byte budget is spent". An empty `prefetch.offload_params` means "offload every
parameter **of each offloaded layer**". The first is bounded by bytes and the
second by layer position.

## What the engine refuses, and what it only warns about

The engine refuses a configuration it cannot honour, at startup, before it reads
any model file:

```text
server: fatal: offload_num_in_group (5) must be <= offload_group_size (4)
```

Refusals cover a malformed document, an unknown backend name, a field of the
wrong type, a negative budget, `offload_num_in_group` above `offload_group_size`,
and `offload_prefetch_step` below 1 while prefetch is enabled.

### The model has to claim support, and none does yet

A valid configuration is still refused when the resolved architecture's loader
does not consult the offloader. This is the refusal you will actually hit, since
**no model declares support today**:

```text
weight offload is configured but architecture "Qwen3MoeForCausalLM" does not
support it: its loader does not consult the weight offloader, so every weight
would stay on the device and the budget would free nothing
(ENG-WEIGHT-OFFLOAD W2c). Remove --offload-config, or wire this model's loader.
```

It is raised after the architecture resolves and **before any weight I/O**, so
nothing is read from disk before you are told. The support flag defaults to off,
which means a newly added model is refused until someone wires its loader —
the default is the mechanism, not an oversight.

A model that *claims* support and then never asks the offloader about a single
weight is refused too, after load, and reported as a defect in that loader
rather than as a configuration error. Zero consulted weights is the only count
that can prove that particular lie.

A backend that disagrees with the fields you set is a **warning**, not a refusal,
which mirrors vLLM. The named backend wins and the other fields are ignored:

```text
[vllm.cpp] offload_config: Prefetch offload fields are set but offload_backend='uva'. Prefetch settings will be ignored.
```

## Why the budget can overshoot by one weight

The engine tests the budget **before** each weight, never against that weight's
size, which is what vLLM does. A 10 GiB budget can therefore be exceeded by the
size of one weight. Treat `cpu_offload_gb` as the point at which offloading
stops, not as a hard ceiling on host memory.

## Limitations

- **No weight moves yet.** See [What works today](#what-works-today).
- **On unified memory the feature cannot help.** On a device such as GB10 the
  host and the device draw on one physical pool, so moving a weight to "CPU"
  frees no device memory. Offload is useful on a discrete GPU, where host memory
  and device memory are separate.
- **The `prefetch` backend is not built.**
- **Offload trades throughput for capacity.** A weight that is not on the device
  is read across the bus when it is needed. Expect a model that fits without
  offload to run faster without it.

## Consuming it programmatically

```c
vllm_model_params p = vllm_model_params_default();
p.model_path = "/path/to/model";
p.offload_config =
    "{\"offload_backend\":\"uva\",\"uva\":{\"cpu_offload_gb\":10,"
    "\"cpu_offload_params\":[\"experts\"]}}";

vllm_engine* engine = NULL;
if (vllm_engine_load(&p, &engine) != VLLM_OK) {
    fprintf(stderr, "%s\n", vllm_last_error());
}
```

A configuration error returns `VLLM_ERR_INVALID_ARGUMENT` and sets
`vllm_last_error()`. That is a different result from `VLLM_ERR_MODEL_LOAD`, which
means the configuration was accepted and the model itself failed to load.
