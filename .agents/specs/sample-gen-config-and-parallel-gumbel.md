# `SAMPLE-CORE` — the checkpoint's sampling defaults, and the Gumbel draw that scans a 248,320-wide vocab on one thread

Issues: [#1985](https://github.com/mudler/vllm.cpp/issues/1985) (checkpoint
sampling defaults are discarded),
[#1984](https://github.com/mudler/vllm.cpp/issues/1984)
(`RandomSampleKernel` is `<<<n, 1>>>`).
Owning row: `SAMPLE-CORE` ([engine matrix](../engine-matrix.md)).
Lifecycle: `ACTIVE`.
Oracle: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` (0.26.0.dev0), the
primary pin, read from the local checkout at that exact SHA.

## Why the two are one row

They are the same failure seen from two ends. On `Qwen/Qwen3.8-27B` vLLM draws
its next token from 20 candidates and we draw ours from 248,320, because we
never read the `top_k` the checkpoint ships. That is a correctness divergence on
the workload the parity gate runs, and it is also why the kernel that does the
drawing has four orders of magnitude more work than it needs. Reading the
config shrinks the *distribution*; it does not shrink the *scan*, because both
engines still walk the whole vocabulary. So the scan has to be parallelised as
well, and the two changes have to be measured together or each will be credited
with the other's effect.

## Scope

1. Parse `generation_config.json`'s sampling keys, mirror
   `ModelConfig.get_diff_sampling_param`, and resolve them in
   `to_sampling_params` with upstream's precedence.
2. Replace the single-thread `RandomSampleKernel` on CUDA with the two-pass
   grid-strided argmax reduction that already sits eleven lines above it, with
   **bit-identical output**.

Out of scope, and named so a later measurement does not credit this row with
them: the `DeviceBuffer probs` per-step `cudaMalloc`/`cudaFree`, the blocking
`rs.download`, the all-greedy gate on the async fast path, the ROCm twin of the
kernel, and the f64-vs-f32 Gumbel dtype divergence. All are recorded under
`## Owed` with anchors.

## Upstream anchors

Read at `555967922` from `${VLLM_SOURCE}`.

| What | Upstream |
|---|---|
| `generation_config` defaults to `"auto"` | `vllm/config/model.py:298` (`ModelConfig.generation_config`) |
| load the file, keep the non-default keys | `vllm/config/model.py::ModelConfig.try_get_generation_config` |
| narrow to six sampling keys, rename `max_new_tokens` | `vllm/config/model.py::ModelConfig.get_diff_sampling_param` |
| the server stores it | `vllm/entrypoints/openai/completion/serving.py:80`; `chat_completion/serving.py:174` |
| the neutral fallbacks | `vllm/entrypoints/openai/completion/protocol.py::CompletionRequest._DEFAULT_SAMPLING_PARAMS` |
| request wins, else checkpoint, else neutral | `vllm/entrypoints/openai/completion/protocol.py::CompletionRequest.to_sampling_params` |
| beam search resolves temperature the same way | `completion/protocol.py::CompletionRequest.to_beam_search_params` |
| the exponential/Gumbel draw | `vllm/v1/sample/ops/topk_topp_sampler.py::TopKTopPSampler.forward_native` |
| `probs.div_(q).argmax(-1)` | `vllm/v1/sample/ops/topk_topp_sampler.py::sample_with_exponential_noise` |
| the noise dtype (f32 by default) | `vllm/v1/sample/ops/topk_topp_sampler.py::empty_exponential_noise_like`; `vllm/v1/sample/sampler.py::Sampler.__init__` (`use_fp64_gumbel: bool = False`) |

Local anchors: `src/vt/cuda/cuda_sample.cu::RandomSampleKernel`,
`::ArgmaxPartialKernel`, `::ArgmaxFinalKernel`;
`src/vllm/transformers_utils/hf_config.cpp::ReadGenerationConfigEosIds`;
`src/vllm/entrypoints/openai/protocol.cpp::CompletionRequest::to_sampling_params`.

## The transformers version decides what `to_diff_dict` keeps, and it is not academic

`try_get_generation_config` returns `GenerationConfig.to_diff_dict()`, which
drops every key equal to a bare `GenerationConfig()`'s value. Under
`transformers` 4.x those defaults were `temperature=1.0`, `top_k=50`,
`top_p=1.0`, so a checkpoint shipping `top_k: 50` would have been dropped and a
port that read the JSON literally would diverge. Under the pinned floor
`transformers >= 5.5.3` (`requirements/common.txt:10`) every sampling field of a
bare `GenerationConfig()` is `None`, so **every declared key survives the diff**
and reading the JSON literally is exact.

Measured, not assumed: against the `transformers 5.3.0` on this host,
`GenerationConfig.from_pretrained(dir).to_diff_dict()` on a Qwen-shaped file
returns `{'repetition_penalty': 1.05, 'temperature': 1.0, 'top_k': 20,
'top_p': 0.95}` — nothing dropped. The mirror therefore parses the JSON
directly, and this paragraph is the reason it is allowed to. If the pin's
transformers floor ever moves back below 5.x, this row's parse becomes wrong and
has to grow the default table.

`Qwen/Qwen3.8-27B`'s file, read live 2026-08-26 from
`https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/generation_config.json`:
`{"temperature": 1.0, "top_k": 20, "top_p": 0.95}` plus the token ids. Our
resolved values today are `temperature 1.0, top_k 0, top_p 1.0` — both filters
disabled.

## Design — part 1, the sampling defaults

- `include/vllm/config/generation.h` + `src/vllm/config/generation.cpp`:
  `DefaultSamplingParams` (the six resolved optionals, `max_new_tokens` already
  renamed to `max_tokens`) and `GetDiffSamplingParam(const HfConfig&,
  const std::string& generation_config)`, where the selector takes upstream's
  three forms: `"auto"` (the checkpoint's own file, the default), `"vllm"` (no
  file, neutral defaults), or a directory holding a `generation_config.json`.
- `HfConfig` grows `generation_config_sampling`, filled by the same sibling read
  that already produces `generation_config_eos_ids`. One file read, two
  consumers.
- `to_sampling_params` and `to_beam_search_params` on both request types take a
  `const DefaultSamplingParams*`; `nullptr` reproduces today's behaviour byte for
  byte, which is what keeps every existing caller and test unchanged.
- Both serving handlers gain `set_default_sampling_params`, called from
  `server_main.cpp` from `loaded->config()` and the new `--generation-config`
  flag, and logged on startup the way upstream logs it.

Precedence, mirrored exactly: an explicitly sent request field wins; an omitted
field takes the checkpoint value; if the checkpoint does not declare it, the
neutral OpenAI default. `"vllm"` restores today's behaviour on demand.

## Design — part 2, the parallel Gumbel draw

`score(row, j) = probs[row][j] / ExpNoise(seed, row, j)` and the answer is
`argmax_j score` with the lowest index winning a tie — which is exactly the
operator `ArgReduce` already implements for greedy argmax, and `ArgReduce` is
**order-independent** (it compares the true global index, not thread or block
order). So the same two-pass partition can carry the Gumbel score with no
change to what is selected.

This is the load-bearing property of the whole change, so state it plainly:
every element's `score` is computed by the identical expression on the identical
device libm, and only the order in which those identical floats are combined
changes. The new kernel is therefore **bit-identical to the old one**, not
merely close, and the gate below asserts equality rather than agreement.

- `include/vt/sample_common.h` (new): `SplitMix64`, `ExpNoise`, `ArgReduce`,
  `kArgSentinel` and `ArgBlocksPerRow` as `__host__ __device__` inlines. Today
  `SplitMix64`/`ExpNoise` are written out three times — `cpu_sample.cpp`,
  `cuda_sample.cu`, `rocm_sample.hip` — and a divergence between any two of them
  is silent. The CPU and CUDA copies are replaced by the header; ROCm is left
  alone deliberately (see `## Owed`).
- `cuda_sample.cu`: the partial kernel is templated on a score functor, so
  greedy argmax and the Gumbel draw share one reduction rather than growing a
  second hand-written copy. Greedy's instantiation is the same source it runs
  today.
- The legacy serial kernel stays, reachable as `VT_FAST_RANDOM_SAMPLE=0`,
  mirroring the `VT_FAST_ARGMAX` lever the greedy rewrite kept. It exists so the
  equality gate is a **same-binary A/B**, which `AGENTS.md` requires before any
  performance result is accepted.
- The Gumbel partials get their own persistent scratch rather than sharing the
  argmax one, because a mixed greedy/random batch runs both in one `sample()`
  call and sharing would make correctness depend on stream ordering that nothing
  in the type system enforces.

## Tests

| Gate | Where | Runs on |
|---|---|---|
| checkpoint `top_k`/`top_p`/`temperature` reach `SamplingParams` when the request omits them | `tests/vllm/test_openai_protocol.cpp` | CPU |
| an explicit request value still wins over the checkpoint's | same | CPU |
| a checkpoint key the JSON does not declare falls to the neutral default | same | CPU |
| `"vllm"` selector discards the file; a path selector reads another directory | `tests/vllm/test_hf_config.cpp` | CPU |
| `max_new_tokens` is renamed to `max_tokens` | same | CPU |
| the sibling parse keeps `eos_token_id` behaviour unchanged | same | CPU |
| `ArgReduce` is order-independent, including on ties | `tests/vt/test_ops_sample.cpp` | CPU |
| the two-pass partition over the production score and reduce equals the serial CPU reference over vocab 1/2/255/256/257/1000/248320, uniform, one-hot, all-equal (all ties), and top-k-masked rows | same | CPU |
| CUDA parallel == CUDA serial **exactly**, same binary, same shapes | same, `HasCuda`-guarded | GPU |
| the server wires the defaults (reachability) | `tests/vllm/entrypoints/test_server_defaults.cpp` | CPU |

## Gates

```sh
scripts/agent-preflight.sh
cmake --build build -j"$(nproc)" && ctest --test-dir build --output-on-failure
```

## The measurement this row does not get to skip

`AGENTS.md` and [#1975](https://github.com/mudler/vllm.cpp/issues/1975) between
them settle it: a performance change here does not merge on a green compile, and
a per-kernel figure is not evidence. [#1929](https://github.com/mudler/vllm.cpp/issues/1929)
landed a 708 us -> 40 us top-k kernel with fresh review, mutation proofs and a
green CUDA build, and cost 16.9 tok/s end to end.

So the accepting evidence is **end to end, on the leased box, run by the
operator**, and it is stated here before it is run. See `## Now` for the exact
request and the predicted values.

## Owed

- [#1984](https://github.com/mudler/vllm.cpp/issues/1984) also names two
  adjacent costs this row does not fix: `src/vllm/v1/sample/sampler.cpp:344`
  allocates the `[n, vocab]` probs buffer through `Backend::Alloc`/`Free`
  (`sampler.cpp:41,47` -> `src/vt/cuda/cuda_backend.cu:80-85`), a 30.3 MiB raw
  `cudaMalloc` plus a device-synchronising `cudaFree` every decode step at B=32;
  and `sampler.cpp:358` `rs.download(...)` is a blocking `Synchronize` every
  step, because the zero-sync device-resident path at `sampler.cpp:456-461`
  requires `sm.all_greedy` and is structurally unreachable at temperature 1.0.
- `src/vt/rocm/rocm_sample.hip::RandomSampleK` is launched `<<<n, 1, 0, stream>>>`
  and carries its own copy of `ExpNoise` (`rocm_sample.hip:38`). It is the same
  defect as #1984 on a backend this row has no hardware to gate, so it is left
  untouched rather than changed blind.
- Our `ExpNoise` computes `-log(u)` in **double** unconditionally. Upstream's
  default is f32 (`use_fp64_gumbel: bool = False`), so this is an unannotated
  widening of exactly the kind `.agents/porting.md` says a token gate cannot
  see, and on a part with 1:64 f64 throughput it is also the dominant remaining
  cost of the parallel kernel. Narrowing it changes which token is drawn, so it
  is a separate row with its own gate and not a rider on this one.
- `--override-generation-config` (`vllm/config/model.py:305`), the
  `override_max_tokens` server-wide output cap derived from `max_new_tokens`
  (`completion/serving.py:81-86`), and an offline equivalent of
  `LLM.get_default_sampling_params` (`vllm/entrypoints/llm.py:404`) for the C
  ABI, which receives explicit `vllm_sampling_params` and has no "omitted"
  state to fill.

## Now

Implementation and gates land together in one pull request with this spec (the
`AGENTS.md` default; no split case applies).

**The end-to-end measurement requested of the operator, with its prediction,
before it is run.**

Workload: `vllm bench serve` against our server and against the pinned vLLM on
the identical `Qwen3.8-27B` artifact, vLLM's production (graphed) configuration
as the denominator, **no `--temperature` flag on either side**, so both engines
resolve temperature 1.0 and both take the random-sampling path. This is the
configuration that made the divergence real, so it is the configuration that
has to judge the fix.

Four arms, one binary, so each half of the row is attributable on its own:

| arm | `--generation-config` | `VT_FAST_RANDOM_SAMPLE` | isolates |
|---|---|---|---|
| A (today) | `vllm` | `0` | the pre-change baseline |
| B | `auto` | `0` | the config read alone |
| C | `vllm` | `1` | the kernel alone |
| D (shipping default) | `auto` | `1` | both |

Predicted, and marked as prediction: **D >= C > A**, with C - A the large term.
The serial kernel's cost is INFERRED from this file's own recorded anchor — a
single-thread scan of a ~151k vocab at ~7.5 ms/token — scaled by 1.64x vocab and
by one f64 `log` plus two 64-bit mixes per element, giving of order 12-20
ms/step at B=32. Parallelising it should leave a sampler bounded by f64 `log`
throughput rather than by one lane, so the predicted recovery is **most of that
per-step cost**, i.e. of order 10 ms/step, and the prediction is stated as a
throughput floor rather than a ratio because the decode step's other terms are
not measured here. B - A is predicted **small and possibly negative** on
throughput: top-k 20 does not reduce the work either engine does, and the reason
to want it is that it makes the two engines sample the same distribution.

Two things would falsify the design rather than the tuning, and both are worth
naming in advance: D materially slower than C would mean the top-k path costs
more than the distribution it saves, and D no faster than A would mean the
sampler was never the bottleneck and the per-step term lives in the
`cudaMalloc`/`cudaFree` and the blocking download recorded under `## Owed`.

`compute-sanitizer` on the new kernel is **requested**, not optional, over the
CUDA equality gate: the change adds a second persistent device scratch and a new
grid geometry, and [#1958](https://github.com/mudler/vllm.cpp/issues/1958) is an
illegal memory access on this same sampler surface.
