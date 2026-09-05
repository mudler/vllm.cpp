# Using vllm.cpp

Use this page for the common ways to run vllm.cpp. Model-specific commands and
specialized tasks have separate indexes below.

## Before you run a model

Build vllm.cpp before you use these commands. See [Building
vllm.cpp](BUILD.md) for CPU, CUDA, Metal, Vulkan, ROCm, and Tenstorrent build
instructions.

The examples use `/path/to/model` for a local model directory. Replace that
path with a compatible checkpoint for the workflow you select.

## Run a local completion

Run one completion with `vllm-cli`:

```sh
build/examples/vllm-cli \
  --model /path/to/model \
  --prompt "The capital of France is" \
  --max-tokens 64
```

Run `build/examples/vllm-cli --help` to list the flags in your build.

`--repeat N` loads the model once and runs N completions, which is how a warm
decode rate is read off this client. It writes two lines to standard error per
completion. The first carries the result and the timing:

```text
vllm-cli: run=2/5 finish_reason=length prompt_tokens=5 completion_tokens=64 secs=1.234 tok_s=51.863
```

The second carries the wall-clock instants that completion generated between, as
Unix epoch seconds:

```text
vllm-cli: run=2/5 generate_start_unix=1755000000.500000 generate_end_unix=1755000006.250000
```

Those instants are what lets a benchmark attribute an out-of-process measurement
-- a GPU clock sampler, a power meter, a profiler -- to the generation rather
than to the whole process, which for a large checkpoint is mostly the load. Both
lines go to standard error, so a pipeline reading the completion off standard
output is unaffected.

## Start the OpenAI-compatible server

Start the server with a local model directory:

```sh
build/examples/vllm-server \
  --model /path/to/model \
  --port 8000 \
  --max-num-seqs 32
```

On a hybrid model -- one that interleaves linear-attention (GDN/Mamba) layers
with full-attention layers, such as the Qwen3.5 and Qwen3.6 families --
`--max-num-seqs` is a ceiling rather than a promise. Each concurrently served
sequence owns one recurrent state per linear-attention layer, and under
speculative decoding it owns `num_speculative_tokens + 1` of them, so the engine
bounds the number of seats by what the KV pool holds and reports any reduction
on standard error:

```text
INFO recurrent-state budget: reduced max_num_seqs from 32 to 13. The KV pool
(3072 blocks) holds 118 unified pages of 832 tokens (one page = one 3371008-byte
GDN state), and each sequence owns 9 of them. Raise --num-blocks /
--kv-cache-memory for more concurrent sequences, or lower
num_speculative_tokens.
```

Raise `--num-blocks` or `--kv-cache-memory` to buy more seats, or lower
`num_speculative_tokens`. A model with no recurrent state is never reduced.

A second, independent ceiling comes from the MODEL rather than from the pool. A
port whose forward serves one sequence per step declares it, and the engine
clamps `--max-num-seqs` to 1 and says so:

```text
INFO model concurrency: reduced max_num_seqs from 7 to 1. This architecture's
forward serves ONE sequence per step and refuses a batched one, and an
EngineCore that meets that refusal dies rather than degrades.
```

The clamp is not tuning. A forward that refuses a batched step throws from
inside the EngineCore loop, and that loop treats a throw as fatal: without the
clamp the first pair of overlapping requests ends the engine and every request
after it returns a 500. Clamped, the same server answers those requests one
after another. Concurrent clients are still accepted; they are served in
sequence. Today `Qwen4ExpForConditionalGeneration` (`qwen4exp`) is the only
architecture that declares it — see the model table in
[FEATURES.md](FEATURES.md) for exactly what that architecture serves.

Send a completion request from another terminal:

```sh
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model","prompt":"The capital of France is","max_tokens":64}'
```

The server also supports OpenAI clients that use
`http://localhost:8000/v1` as their base URL. The model-specific guides record
extra files and launch flags when a model needs them.

`/v1/chat/completions` renders the checkpoint's own chat template, and it takes
`chat_template_kwargs` for the extra Jinja variables a template gates on, the
same field and the same name vLLM uses:

```sh
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model","messages":[{"role":"user","content":"hi"}],
       "chat_template_kwargs":{"enable_thinking":false}}'
```

A key you do not send is not a template variable at all, so a template asking
`{% if enable_thinking is undefined %}` gets its own default: the Qwen3.8 family
reasons unless you turn it off. `--enable-thinking` and `--no-enable-thinking`
set the server-wide default, and a request's own keys win over them. Passing
neither flag is not the same as `--no-enable-thinking`; it leaves the variable
unset, which is what vLLM does. [Server reference](reference/server.md) carries
the endpoint and flag tables.

`--model` also takes a Hugging Face repository name, which the server fetches
into the cache before it binds:

```sh
build/examples/vllm-server --model Qwen/Qwen3-0.6B --port 8000
```

That form needs a build that carries transport layer security. The default
`-DVLLM_CPP_OPENSSL=ON` is the tested path, and every release lane and every
container image uses it; `-DVLLM_CPP_BUILD_BORINGSSL=ON` is offered and has
never been compiled here. A build that mixes the two states across its own
source files refuses to start with exit 2 and a message naming what disagrees,
rather than serving corrupted responses. See [Access Hugging Face
checkpoints](guides/hugging-face-access.md) for the build options,
`--revision`, `--download-dir`, the `HF_*` environment variables, and the
release lanes that carry no fetch. `vllm-cli` and the C ABI still take a local
path only.

That command is measured, not illustrative. On 2026-08-20, on x86_64 with the
default OpenSSL build and an empty `HF_HOME`, it fetched
`Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca` from
`huggingface.co`: `model.safetensors` (1503300328 bytes), `tokenizer.json`
(11422654), `vocab.json` (2776833), `merges.txt` (1671853),
`tokenizer_config.json` (9732), `config.json` (726) and
`generation_config.json` (239), 1.5 GB of cache in total. The server then bound
its port and answered `/v1/completions`. A second start with the same `HF_HOME`
reports every file as `already in the cache` and transfers no bytes. Before
[#1511](https://github.com/mudler/vllm.cpp/issues/1511) this command downloaded
nothing at all, because the hub answers with a relative `Location` header that
the client read as a URL.

### Read the model's own distribution with `prompt_logprobs`

Both generation endpoints take `prompt_logprobs`, the same field and the same
name vLLM uses. It scores the prompt you sent: for every prompt position after
the first, the response carries the distribution the model assigned to the token
that actually follows, plus that position's top alternatives. Nothing is
generated to obtain it, so this is how you compare two engines on the same
trajectory rather than on whatever each one decides to say next.

```sh
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model","prompt":"The capital of France is",
       "max_tokens":1,"prompt_logprobs":5}'
```

`choices[0].prompt_logprobs` is then an array with one entry per prompt token.
The first is `null`, because the first token has nothing before it to predict
it; every later entry maps a token id to `{"logprob", "rank", "decoded_token"}`,
where `rank` is the 1-based vocabulary rank and rank 1 is the position's most
likely token. `/v1/chat/completions` takes the same field and returns the array
as a **top-level** `prompt_logprobs` on the response, not on a choice, because
one rendered prompt is shared by every choice. Both match vLLM.

`prompt_logprobs: -1` asks for the whole vocabulary at every position. vLLM
refuses that request unless `--max-logprobs` allows it; this server has no
separate `max_logprobs` and caps at the vocabulary size, so `-1` is served here.

Three request shapes are refused with `400`, as vLLM refuses them:
`prompt_logprobs` together with `"stream": true` (the payload cannot be framed
into the stream), a negative value other than `-1`, and a non-numeric value.

`logprobs` (completions) and `logprobs` + `top_logprobs` (chat) work as they do
in vLLM and score the GENERATED tokens instead.

### Halve the KV cache with `--kv-cache-dtype fp8`

Store the paged K/V as 1-byte fp8-e4m3 instead of 2-byte bf16. The KV block
halves, so the same memory budget holds twice the context:

```sh
build/examples/vllm-server \
  --model /path/to/model \
  --kv-cache-dtype fp8 \
  --kv-cache-memory 8589934592
```

Values are vLLM's own `CacheDType` names. `auto` is the default and uses the
model dtype. `fp8` and `fp8_e4m3` select the quantized store; `bfloat16` names
the default storage dtype explicitly. `float16` and `fp8_e5m2` parse and are
then refused by name, because no attention block writes either yet.

**The checkpoint can ask for it.** When you pass no flag, the server reads the
checkpoint's `config.json` `quantization_config` (falling back to a standalone
`hf_quant_config.json`, which is what ModelOpt 0.29.0 and before wrote) and
honours a declared `kv_cache_quant_algo`, printing one line naming what it
resolved. An explicit `--kv-cache-dtype` always wins over the declaration. Both
the order of the two files and the precedence mirror vLLM.

Check which document your checkpoint declares in before you rely on this. A
repository can carry a current `config.json` beside a stale
`hf_quant_config.json` that disagrees with it, and the inline one is the one
that counts — on this server and on vLLM. `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`
is exactly that shape: only its legacy file mentions the KV cache, so neither
engine turns fp8 KV on for it and the flag has to be typed.

Note that `--kv-cache-memory` is what turns the halved block into twice the
pool. Without it the server falls back to a fixed block count, and `fp8` then
halves the KV bytes for the same context instead.

**`--kv-cache-memory` now bounds the whole pool, and it did not before.** The
value is an absolute budget for the paged KV cache, and the engine sizes the
block count so that everything it allocates fits inside it — which is what vLLM
means by the flag. Until #1963 the divisor counted one layer per KV group while
the engine allocated a buffer per layer, so the same number bought as many times
the memory as the model has attention layers: 8.5 GiB of buffers for
`--kv-cache-memory 1073741824` on the 27B. If you tuned this flag against the
old behaviour, the same value now gives a shorter served context; raise it, and
the auto-fit line on stderr tells you what it settled on.

`--num-blocks` is unaffected: it names a per-layer block count and always did,
so a launch line that sizes the pool that way means exactly what it meant
before. Only the byte budget converts differently. The recurrent-state clamp
(#1983) reads the resolved block count, so at a fixed `--kv-cache-memory` it
now seats fewer concurrent sequences than it did — it is being told the pool's
true size for the first time, and the `INFO recurrent-state budget:` line names
what it compared.

**It costs you the fast attention kernels, and we have not measured the net.**
An fp8 KV cache is read by the tiled prefill and block decode kernels only.
FA-2 prefill, all three FA-2 decode topologies, the WMMA ladder and the
vectorized decode-opt/GQA kernels are bf16-native and are skipped whenever the
cache is not bf16. So this flag buys half the KV bytes and twice the pool, and
spends an unmeasured amount of attention throughput to do it. Which way the sum
goes depends on your model, context length and concurrency; measure your own
workload both ways rather than assuming the memory win is free.

**Accuracy.** A checkpoint that declares fp8 KV but ships no `k_scale`/`v_scale`
tensors serves on the default scale 1.0, and the server says so on stderr. That
is the documented default, not a silent one — and a checkpoint that declares
nothing never reaches it.

**Coverage.** The store and the scaled read are routed for the Qwen3.5/3.8
family and for the shared dense-attention seam, which serves Qwen3 dense,
Qwen3-MoE, Voxtral and the Llama, Mistral and InternLM2 registries. The other 16
architectures carry their own attention preamble and refuse before writing
anything, rather than writing floats into a half-sized block. Only one of them
(Nemotron-H) tells you what you asked for: its refusal names the fp8 KV scheme.
Qwen3-VL reaches the store, which names the op that should have been called and
says the architecture is not routed for fp8 KV. The other 14 report a dtype rule
instead — 13 say `"<arch>: KV cache must be bf16 or f32"`, and Gemma-4 dies one
step earlier inside a cast with `"cast_f32: out must be f32"`, which does not
even name the architecture. Every one of the 16 refuses before writing, so the
half-sized block is never fed floats; what differs is how much the message tells
you. Metal and ROCm refuse it too. See
[the row spec](../.agents/specs/fp8-kv-cache.md) for the exact list.

A refusal arrives AFTER the pool has already been sized at half, which is the
intended order: the sizing is what a wrong answer would corrupt silently, so it
is made consistent first and the unrouted store then says so out loud. On a
heterogeneous-KV model such as Gemma-4, where each layer carries its own
attention spec, that means you see a doubled block count in the startup line and
then a named refusal at the first forward — not a served run.

**Through the C ABI (v24).** `vllm_model_params.kv_cache_dtype` (ABI v24,
fork issue #7) carries the same string the server flag takes. NULL or `"auto"`
is the default and uses the model dtype — byte-identical to before the field
existed. `"fp8"` or `"fp8_e4m3"` stores 1-byte fp8-e4m3 K/V:

```c
vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/path/to/model";
mp.kv_cache_dtype = "fp8";
vllm_engine *engine = NULL;
vllm_engine_load(&mp, &engine);
```

`vllm-cli` takes the same `--kv-cache-dtype` flag the server takes.

`vllm-bench` takes it too, and its report header names the KV dtype it measured
on -- both the value you asked for and the storage dtype the loader actually
sized blocks from. Those two differ when the checkpoint declares
`kv_cache_quant_algo` and you typed no flag, which is exactly the case where a
published number would otherwise be unattributable:

```text
KV cache dtype (requested):                auto
KV cache dtype (resolved storage):         fp8_e4m3 (1-byte pages)
```

## Draft with a second checkpoint

Speculative decoding runs a small draft model beside the target and verifies its
proposals losslessly, so the emitted tokens do not change. Pass the draft with
`--speculative-config`:

```sh
build/examples/vllm-server   --model /path/to/target   --speculative-config '{"method":"dflash","model":"/path/to/draft","num_speculative_tokens":7}'
```

The draft may be a checkpoint directory or a single `.gguf` file, for DFlash,
DFlash2 and DSpark alike. A GGUF draft is dequantized to bf16 as it loads, so
picking a smaller quantization saves download and disk and does not save memory.

**The TARGET's `lm_head` may be quantized.** A DFlash or DFlash2 draft owns no
output head and runs the target's, so until
[#1628](https://github.com/mudler/vllm.cpp/issues/1628) that head had to be stored
as dense bf16: pointing a draft at a safetensors target whose `lm_head.weight` is
ModelOpt or compressed-tensors NVFP4 refused the load with `dflash: target tensor
lm_head.weight is not BF16 (got U8)`. It is now kept packed and multiplied by the
same GEMM the target's own logits take. A head this engine could only read by
WIDENING it still refuses by name -- a GGUF target's `output.weight`, an FP8
`lm_head`, and an NVFP4 head under `VT_MODELOPT_W4A4=1` -- because the DFlash2
candidate selector reads the target head's exact top-K and a widened head changes
that set with no visible symptom. A DSpark draft still refuses every quantized
target head. `VT_LMHEAD_FP4=0` (see [environment](ENVIRONMENT.md)) rolls the
packed head back for the whole engine, and it rolls this refusal back with it:
the draft load then fails by name on an NVFP4 target, which is the pre-#1628
behaviour and is the point of a rollback.

Loading a DFlash2 draft prints a notice to **stderr**, on both the safetensors
and the GGUF arm. **It prints TWICE per load**, on the server, the C API and the
bench client alike: the loader reaches the same check from two places on one set
of engine parameters — directly, before the target is mapped, and again through
the speculative-config resolution the engine constructor runs — and the check
carries no once-flag. That is a known defect and it is cosmetic: nothing is
refused, and no WEIGHTS are loaded twice -- what re-runs is the classification
and its paragraph
([#1607](https://github.com/mudler/vllm.cpp/issues/1607)). The notice is purely
informational. It names what runs, what is still owed (the bf16 residency
above, and that no throughput number has been taken), and that the port mirrors
[vllm#52816](https://github.com/vllm-project/vllm/pull/52816), which merged
upstream on 2026-08-21 at `3406ec1d` and onto which this port is not yet
reconciled ([#1561](https://github.com/mudler/vllm.cpp/issues/1561)).

[Speculative decoding](SPECULATIVE-DECODING.md) lists the supported methods, the
draft checkpoints each was gated against, and what each one refuses by name.
Drafting is greedy: `draft_sample_method` accepts only `"greedy"`, and any other
value is refused at startup rather than silently ignored.

The same flag also takes one key vLLM does not have, `vllm_cpp.drafter_chain`,
which names several speculators in preference order. It is parsed and checked
today and **refused at startup**, because nothing resolves a chain yet; the same
page says what the document looks like and what each rule refuses.

## Reuse the NVFP4 tactic draw between runs

An NVFP4 W4A4 model runs its general matrix multiplications through CUTLASS, and
the engine picks one tactic for each matrix shape. The first run on a device
times every candidate tactic for a shape and keeps the fastest. The engine
writes the winners to a JSON document, and a later run on the same device and
the same build reads that document instead of timing again.

Read this cache as a reproducibility and warmup control, not as a speed feature.
The measured steady-state component of a frozen plan map on the GB10 lane is
1.0045x at concurrency 2 and 1.0050x at concurrency 16, and the strict component
result FAILED at 39 of 40 timing axes and 1 of 8 memory axes. [The row
spec](../.agents/specs/nvfp4-persistent-plan-cache.md) records that run. What
the cache gives you is that two runs of the same binary on the same device start
from the same kernel plan, and that a run which loaded a document does not pay a
tuning pass. A throughput claim beyond that needs its own measurement.

The cache covers one process on one device. Tensor parallelism above one rank
is out of scope for it, and [the row
spec](../.agents/specs/nvfp4-persistent-plan-cache.md) records why.

### The cache is on by default

`VT_FP4_PERSISTENT_CACHE` defaults to on. The engine reads a document if one
matches, tunes the shapes that are missing, and writes the merged result back at
the end of the warmup. Set the variable to `0`, `false`, or `off` to keep the
tuning in memory and write nothing. The engine refuses any other value at
startup with `invalid boolean value for VT_FP4_PERSISTENT_CACHE`.

Without a path override, the document lands under `$XDG_CACHE_HOME`, or under
`$HOME/.cache` when `XDG_CACHE_HOME` is unset. The path names every input that
the document is valid for:

```text
<root>/vllm.cpp/nvfp4_autotune/vllm.cpp_nvfp4_autotune_v1/sm_<architecture>/
  device_<ordinal>-gpu_<device name>/
  cuda_<runtime version>-driver_<driver version>/
  cutlass_<version>/
  tactics_<descriptor digest>-set_<tactic set version>/
  dtype_<output dtype>-id_<dtype id>-fp4_<fp4 layout>-sf_<scale layout>/
  timing_w<warmups>-r<repeats>-d<delay microseconds>-bucket_<bucket version>/
  build_<tactic ABI digest>/autotune_configs.json
```

That is one path. The example wraps it for reading, and the indentation is not
part of it. Every character outside `A-Z`, `a-z`, `0-9`, `.`, `_`, and `-`
becomes `_` in a path component, so a device name such as `NVIDIA GB10` reads as
`NVIDIA_GB10`.

If `XDG_CACHE_HOME` and `HOME` are both unset, the engine has no cache root. It
then turns the cache off for that run and tunes in memory. In read-only mode,
with no imported document, the same condition is refused with
`read-only NVFP4 cache requested without a cache path/root`.

These are the variables that control the cache:

| Variable | Default | What it does |
|---|---|---|
| `VT_FP4_PERSISTENT_CACHE` | on | Off turns the persistent document off for the run. Takes `1`, `true`, `on`, `0`, `false`, or `off` |
| `VT_FP4_AUTOTUNE_CACHE_PATH` | None | Names the native document file and replaces the derived path |
| `VT_FP4_FLASHINFER_CACHE_PATH` | None | Imports a FlashInfer `autotune_configs.json` before the native document loads |
| `VT_FP4_AUTOTUNE_CACHE_READONLY` | off | Loads plans, tunes nothing, and writes nothing. A missing plan is refused |
| `VT_FP4_AUTOTUNE_DELAY_US` | `5000` | Idle microseconds before the timed repeats of one candidate. Maximum `1000000` |
| `VT_FP4_AUTOTUNE_VERBOSE` | off | Prints one `[VT_FP4_AUTOTUNE]` line for each tactic the engine times |

The timing recipe is FlashInfer's. For each candidate tactic the engine runs 3
warmup launches, synchronizes the stream, holds it idle for 5,000 microseconds,
and then takes the mean of 10 event-timed launches.
`VT_FP4_AUTOTUNE_DELAY_US` changes that idle time, and it also changes the cache
path, because the recipe is part of the document identity. The value `0` removes
the delay and keeps the no-delay diagnostic arm.

### Import an existing FlashInfer cache

If you already ran vLLM with the FlashInfer autotuner on the same box, point the
engine at the file that run produced:

```sh
VT_FP4_FLASHINFER_CACHE_PATH=$HOME/.cache/vllm/flashinfer_autotune_cache/0.6.13/121a/<config hash>/autotune_configs.json \
build/examples/vllm-server --model /path/to/nvfp4-model
```

The import reads the file and never writes to it. The engine installs the
imported plans before it reads the native document, so an imported plan wins
over a native one for the same shape.

The import is strict, and a failed import stops the load rather than falling
back. The file's `_metadata` object must declare FlashInfer version `0.6.13`,
cuBLAS version `13.1.0`, cuDNN version `91900`, and cuDNN frontend version
`1.26.0`. Its `cuda_version` must equal the running CUDA runtime version, and
its `gpu` must equal the running device name. The literal value `*` in any one of
those six fields matches anything. The engine reads only the entries whose key
starts with `('fp4_gemm'` and whose value is the pair
`["CutlassFp4GemmRunner", <tactic id>]`. It counts and ignores every other
entry, and it refuses a duplicate `fp4_gemm` key.

### Freeze a draw for a benchmark run

Tuning re-times the candidates in every fresh process, and CUTLASS reduction
order follows the tactic. Two runs that each tuned for themselves can therefore
select different tactics and produce different token ids for the same prompt.
Freeze the draw so that tactic selection is not a variable in your measurement:

```sh
VT_FP4_AUTOTUNE_CACHE_READONLY=1 \
build/examples/vllm-server --model /path/to/nvfp4-model
```

In this mode the engine loads the document, tunes nothing, and writes nothing.
A shape with no plan in the document is refused before the server is ready, with
`NVFP4 frozen persistent cache miss before readiness:` and the M bucket, N, K,
device ordinal, and streaming multiprocessor architecture that missed. Take one
normal read-write run first, so the document covers the shapes your workload
uses, then freeze it.

The tree ships no measured draw, so a fresh checkout tunes its first run.
[Issue #2752](https://github.com/mudler/vllm.cpp/issues/2752) owes a pinned GB10
draw and its install step.

### Confirm the cache was used

The engine prints two `[VT_FP4_CACHE]` lines on stderr. The first reports what it
resolved, before it serves:

```text
[VT_FP4_CACHE] prepared mode=read-write native=<path> flashinfer=<path> loaded=64 (flashinfer=0 native=64) rejected=0 delay_us=5000 metadata=<fingerprint> selected=64
```

The second reports what it published, after the warmup, and one
`[VT_FP4_CACHE] selected M=<bucket> N=<n> K=<k> tactic=<id>` line follows it for
each plan in the map:

```text
[VT_FP4_CACHE] complete mode=read-write loaded=64 tuned=0 rejected=0 saved=64 selected=64 metadata=<fingerprint>
```

Read the fields as follows:

| Field | Meaning |
|---|---|
| `mode` | `read-write` is the default, `read-only` is the frozen mode, and `disabled` means the cache was off. `read-write-native-rejected` and `read-only-native-rejected` mean the document did not match. `read-write-save-failed` means the write did not happen |
| `native` | The resolved native document path. Empty when no path resolved |
| `flashinfer` | The imported FlashInfer file. Empty when you set no import path |
| `loaded` | Plans installed from a document, split into the FlashInfer and native counts |
| `rejected` | Documents refused for a metadata mismatch, not plans |
| `delay_us` | The resolved tuning delay, which is part of the document identity |
| `metadata` | A 16-digit hexadecimal fingerprint of the resolved identity. Two runs that print the same value agree on every keyed input |
| `selected` | Plans in the live map at that moment |
| `tuned` | Shapes this process timed itself |
| `saved` | Plans written to the native document |

A run that reused a document prints `loaded` equal to `selected` and `tuned=0`.
A run that tuned prints `loaded=0` and a non-zero `tuned`. A `rejected=1` with a
non-zero `tuned` means the engine found a document, refused it, and tuned
instead.

Two other lines report a problem by name. A refused native document prints
`[VT_FP4_CACHE] rejected native cache path=<path> error=<reason>` and names what
the engine did next. A shape that misses after the pre-serve warmup prints
`[VT_FP4_AUTOTUNE] lazy-miss after pre-serve warmup` with the shape, which means
your workload reached a shape the warmup did not cover.

### What invalidates a document

The path carries the identity, and the document repeats it in a `_metadata`
object. The engine compares the two and refuses a document that does not
describe the running configuration, rather than serving plans that were measured
somewhere else. A change to any of these values gives a different path and
leaves the old document in place, unread:

- the streaming multiprocessor architecture, the device name, and the device
  ordinal
- the CUDA runtime version and the CUDA driver version
- the CUTLASS version
- the tactic descriptor digest and the tactic set version
- the output dtype, the FP4 layout, and the scale layout
- the timing recipe, which is the warmup count, the repeat count, the delay in
  microseconds, and the M bucket version
- the tactic ABI digest, which is a SHA-256 over the CUTLASS version and the
  tactic source files

A rebuild that changes no tactic source keeps the same tactic ABI digest, so an
ordinary rebuild does not invalidate your cache. A driver update does, and so
does moving the same file to a second GPU.

In the default read-write mode, a refused native document is not fatal. The
engine tunes the run, and it does not overwrite the file it could not read. In
read-only mode with no imported file, a refused document stops the load.

### Tune every shape before the first request

By default the engine tunes the whole hybrid profile set before it serves, using
one internal greedy request of `max_num_batched_tokens` tokens. This moves the
tuning cost out of your first real request. It runs only when the model is NVFP4
W4A4, the device is a CUDA device of compute capability 10.0 to 12.9, and none
of `VT_FP4_PRE_SERVE_WARMUP`, `VT_FP4_AUTOTUNE`, and `VT_FP4_PLAN_CACHE` is set
to a value that starts with `0`. Each of the three defaults to on when it is
unset. Setting any one of them to `0` turns the pre-serve warmup off and moves
the tuning back into the first requests that meet each shape.

## Use the C ABI

For an installed library, use the stable public interface in
[`include/vllm.h`](../include/vllm.h). Link `libvllm` and include `vllm.h`.
This example shows the blocking completion shape:

```c
#include "vllm.h"

vllm_model_params model = vllm_model_params_default();
model.model_path = "/path/to/model";

vllm_engine *engine = NULL;
if (vllm_engine_load(&model, &engine) != VLLM_OK) {
    fprintf(stderr, "%s\n", vllm_last_error());
    return 1;
}

vllm_sampling_params sampling = vllm_sampling_params_default();
sampling.max_tokens = 64;

vllm_completion output;
if (vllm_complete(engine, "The capital of France is", &sampling, &output) == VLLM_OK) {
    printf("%s\n", output.text);
    vllm_completion_free(&output);
}
vllm_engine_free(engine);
```

`vllm_engine_spec_acceptance` (ABI v25) reads back the speculative-decoding
counters the engine already keeps, so a client can say why a throughput number
is what it is. The three counts are CUMULATIVE over the handle's life; subtract
two reads to get a per-request or per-leg figure, which is what `vllm-cli` does
per `--repeat` leg:

```c
vllm_spec_acceptance before, after;
vllm_engine_spec_acceptance(engine, &before);
vllm_complete(engine, "The capital of France is", &sampling, &output);
vllm_engine_spec_acceptance(engine, &after);

int64_t proposed = after.drafts_proposed - before.drafts_proposed;
int64_t accepted = after.drafts_accepted - before.drafts_accepted;
int64_t steps    = after.drafted_request_steps - before.drafted_request_steps;
```

`drafts_accepted` EXCLUDES the bonus/replacement token a verify step always
emits, which is vLLM's own convention for
`vllm:spec_decode_num_accepted_tokens`, so `accepted / proposed` is the
acceptance rate with the bonus token out of it.

`drafted_request_steps` counts (request, step) PAIRS that carried at least one
draft, NOT forward passes: one verify forward over a batch of 8 drafted requests
adds 8, and the two numbers are equal only at `max_num_seqs=1`. That is also how
vLLM's `vllm:spec_decode_num_drafts` and SGLang's `spec_verify_ct` count, so the
figures compare directly at any concurrency.

`1 + accepted / steps` is vLLM's `mean_acceptance_length` EXACTLY, with the bonus
token in it. It is NOT SGLang's `accept_length`, which divides
`completion_tokens` by `spec_verify_ct`: that numerator also carries the
request's prefill token, which came from no verify step, so SGLang's figure runs
about one token per request higher. Quoting one number under both names is an
error of roughly +0.07 on a 64-token request.

All three read 0 on an engine that never speculated.

`vllm-cli` prints the per-leg deltas on stderr, on their own line, beside the
timing and leg-boundary lines it already prints:

```text
vllm-cli: run=2/5 spec_drafts_proposed=196 spec_drafts_accepted=90 spec_drafted_request_steps=28
```

`vllm_chat` takes a whole OpenAI chat request as JSON, so it accepts
`chat_template_kwargs` exactly as the server does, and it applies the same
default: a key nobody sends is not a template variable at all, so a Qwen3.8
checkpoint reasons unless the request turns it off. A key that names something
the renderer supplies (`messages`, `tools`, `chat_template`, `tokenize`) is
refused with `VLLM_ERR_INVALID_ARGUMENT` rather than honoured, so no request can
replace the conversation the caller passed in `messages`.

## Use the internal C++ library in the source tree

The headers under [`include/vllm/`](../include/vllm/) are source-tree
internals. They are not an installed or stable public ABI. Repository targets
can include these headers and link the internal `vllm::vllm` CMake target.

For example, a source-tree target can load a model directory through
`LoadedEngine`:

```cpp
vllm::entrypoints::EngineParams params;
params.enable_prefix_caching = true;
params.policy = vllm::SchedulerPolicy::kLPM;
auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model_dir, params);
```

See [`entrypoints/model_loader.h`](../include/vllm/entrypoints/model_loader.h)
for `LoadedEngine`. The source-tree examples declare their link targets in
[`examples/CMakeLists.txt`](../examples/CMakeLists.txt). External consumers
must use the C ABI in `include/vllm.h`.

Configuring with `-DVLLM_CPP_SANITIZE=address,undefined` or
`-DVLLM_CPP_SANITIZE=thread` changes what a test target links. Instrumented
test executables link one internal shared image of the instrumented archive
instead of force-linking `vllm::vllm` into each of them, because the
force-linked form runs a hosted runner out of disk. That image forwards the
same include directories, compile definitions and link libraries, so a target's
own CMake is the same in both configurations. It does not LINK identically: the
archive is force-linked into each executable only in the default build, and not
propagating that is the reason the instrumented image exists. Link `vllm::vllm`
as above and let the build choose; naming the internal image yourself is not
supported.

## Re-derive a benchmark rather than read one

Two figures in [Benchmarks](BENCHMARKS.md) come from executables that the
ordinary build compiles and that no test runs, so a reader can reproduce them
without a checkpoint. Both allocate hundreds of megabytes and spend tens of
seconds per sweep, which is why CI compiles them and runs neither.

```sh
cmake --build build --target vllm_music3_vocoder_conv_ab vllm_conv1d_scaling_probe

# The MiniMax-Music3 vocoder decode window at the shipped geometry, over a
# sweep of latent window lengths. It prints an FNV-1a fingerprint of the whole
# stereo waveform, so two builds can be compared for BIT identity rather than
# for closeness.
./build/vllm_music3_vocoder_conv_ab --lengths=20,86,344 --repeats=3

# The same window split into its leaves -- conv1d, conv_transpose, snake, pad,
# copy, residual_add, tanh -- through the production instrument.
VLLM_CPP_MUSIC3_PROFILE=1 ./build/vllm_music3_vocoder_conv_ab --lengths=86

# `vt::Conv1d` alone at the vocoder's eleven geometries, with a residency
# sweep, the pool's dispatch cost and CPU-over-wall per leg.
./build/vllm_conv1d_scaling_probe --latents=86 --repeats=2
```

`VLLM_CPP_CPU_THREADS` selects the pool size for both, and both print the
thread count they actually got beside the count that was asked for.

## Run a gate that needs a GPU and a checkpoint

Most of the suite runs anywhere. Two tests cannot:
`test_minimax_music3_device_arm_real` and `test_minimax_music3_depth_arm_real`
each need an accelerator **and** a 28.5 GB checkpoint, so no
continuous-integration runner can execute either. Both carry the CTest label
`gpu;checkpoint;music3` so that they are selectable by name rather than by
whoever remembers they exist, and a missing precondition makes them exit 77,
which CTest reports as **Skipped** rather than Passed.

```sh
ctest --test-dir build -L gpu -N        # list them; expect `Total Tests: 2`
ctest --test-dir build -L gpu -V        # run them
```

**Read the count, not the exit status.** `ctest -L <label>` prints
`No tests were found!!!` and still returns 0 when the label selects nothing, so
a renamed or dropped label reads as a clean run of a gate that never executed.

**`-L gpu` is not a taxonomy of the device gates**, and `-LE gpu` is not
"everything else". Exactly two tests in this tree carry a label today and they
are these. The other checkpoint-gated suites --
`test_minimax_music3_ar_real`, `_llm_real`, `_acoustic_real`, `_quant_real`,
`_e2e_real` and `test_muse_glimmer_real_weights` -- carry no label, and unlike
these two they do not exit 77: without a checkpoint they print a `SKIP` line and
return normally, so **CTest reports them Passed**. For those, read the
transcript rather than the CTest verdict.

Both drive the C ABI with `device = 1` and assert, from the engine's own profile
buckets, which arm ran -- `test_minimax_music3_device_arm_real` for the 2.4B
flow-matching transformer and `test_minimax_music3_depth_arm_real` for the
0.646B RVQ depth decoder. They are separate entries because the two arms are
selected at two separate call sites on one `--speech-device 1` switch, which is
how one of them drifts. Each arm agrees numerically with its host reference by
design, so the audio cannot answer the question and neither gate asks it to.

```sh
# Inside an `rc` lease on a fleet device -- never over `ssh`.
# Stage the checkpoint to LOCAL disk first: read over the shared CIFS mount it
# is the dominant cost of the run.
export VLLM_CPP_MUSIC3_CHECKPOINT=/local/disk/minimax-music3
ctest --test-dir build -R test_minimax_music3_device_arm_real -V
ctest --test-dir build -R test_minimax_music3_depth_arm_real -V
```

Without `VLLM_CPP_MUSIC3_CHECKPOINT` both gates fall back to
`${CHECKPOINT_ROOT}/minimax-music3`, and without either they skip and say so.
They need a build configured with an accelerator backend; on a CPU-only build
`--speech-device 1` is refused by name before a queue exists, and each gate
skips with that refusal quoted.

## First-line troubleshooting

- Run the executable with `--help` and confirm that you are using the expected
  build directory.
- Check [Environment variables](ENVIRONMENT.md) for settings that can override
  command-line or configuration values.
- Check [Features](FEATURES.md) for the current backend and model surface.
- Read the matching model or task guide before you add model-specific flags.
- If startup fails, use the exact error text to find the refused file, option,
  operation, or checkpoint arm in the focused guides.
- On ROCm, see [Q8_K activation quantization](ROCM.md#select-q8_k-activation-quantization)
  to inspect or override the architecture-scoped quantizer during
  troubleshooting.
- On ROCm, GGUF mixture-of-experts checkpoints compute on the quantized
  expert blocks (Q8_0, Q4_K, Q5_K, Q6_K) instead of being dequantized to
  bf16 at load time.
- On ROCm, mixture-of-experts models run the shared-expert gate and both
  expert-combine steps on device. Before these ops were registered the
  engine refused with `no kernel for op` on that path.
- An architecture whose forward does not read the asynchronous runner's DEVICE
  token identifiers is REFUSED on a step whose host identifiers are stale, with a
  message naming the architecture and `consumes_device_token_ids`. Before this
  refusal the same step ran and generated from token id 0 at `rc=0`, producing
  fluent wrong output rather than an error. It fires only when the runner
  actually spliced a sampled token into one of the step's rows, so prefill-only
  work — including every pooling and embedding request — is unaffected.
  `VT_ASYNC_DEVICE_MIRROR=0` is the same-binary rollback: it returns the host
  combine, makes the host identifiers authoritative, and leaves the refusal
  inert. See [Environment variables](ENVIRONMENT.md).
- Some architectures cannot be paged at the default KV block size of 32, and the
  engine RAISES it for them rather than refusing. DeepSeek-V4-Flash needs 256,
  because a `compress_ratio`-128 layer stores `block_size / compress_ratio`
  tokens per page and 32 gives zero. The number is read from the model, not from
  a flag, so `vllm-cli` -- which has no `--block-size` -- serves these models as
  well as `vllm-server` does. A `--block-size` smaller than the model's floor is
  raised to it and a line on stderr says so; a larger one is kept. Models with no
  such constraint are unaffected.
- A KV-cache block size that is not a multiple of 16 is refused while the
  engine SELECTS an attention backend, with `No valid attention backend for
  device type ...` naming each candidate and `block_size not supported`. On
  ROCm this refusal used to arrive later and read `Block size must be a
  multiple of 16.`, because `ROCM_ATTN` advertised block sizes its cache
  allocation then rejected. Every device now refuses at the same point with the
  same message.
- On ROCm, decode-shaped GEMMs (batch of 4 or fewer, bf16) run on a split-K
  skinny-GEMM kernel rather than the tiled BLAS path. Set `VT_ROCM_SKINNY=0`
  to restore the BLAS path when you want to compare the two.
- On ROCm, Gemma-4 FP8 mixture-of-experts decode uses the device-indexed
  expert gate for batches up to 63 tokens; wider batches use the
  prefill-batch path. Set `VT_GEMMA4_DECODE_INDEXED_MAX_T=1` to restore the
  previous single-token gate when you want to compare the two paths. See
  [Environment variables](ENVIRONMENT.md).
- `--speech-device 1` REFUSES by name instead of falling back to the CPU. It
  refuses when the build registers no accelerator backend, and separately when
  the platform it resolves declines the speech family because that backend is
  partial; the message says which of the two it is. One flag places every stage
  a family can move -- for MiniMax-Music3 the language model, the RVQ depth
  decoder and the flow-matching transformer -- and there is no per-stage switch
  and no environment variable that turns one of them on by itself.
- `nemotron_h`, `laguna` and `qwen3_vl` finish their forward on the host and
  hand the runner a host logits buffer, while the sampler itself runs on device
  (`scripts/runner-routing-allowlist.txt` lists them and names what removes each
  entry). On a unified-memory device — GB10 and other integrated CUDA devices,
  integrated Vulkan, and CPU — those logits are sampled where they are. On a
  discrete GPU they are staged into device memory once per step, into a buffer
  that is reused and only ever grows, so you pay one host-to-device transfer of
  `rows x vocab x 4` bytes per step on these three models and on no others.
  Before [#1313](https://github.com/mudler/vllm.cpp/issues/1313) the host
  address was handed to the sampling kernel directly, which is valid only on a
  unified-memory device; an illegal-address abort during sampling on a discrete
  GPU on an older build was this. The discrete arm is gated at the seam
  (`tests/vllm/v1/sample/test_host_buffer_staging.cpp`) and has no hardware run
  behind it, because every GPU on the project's fleet reports unified memory.
- `tokenizer: merge token "..." at merge rank N ... is not in the vocabulary`
  means the tokenizer file names a merge whose left token, right token, or
  joined result is missing from its own vocabulary. Both `tokenizer.json` and a
  GGUF's `tokenizer.ggml.merges` are checked, and the message names the missing
  token. HF `tokenizers` refuses the same file for the same reason, so the file
  is malformed rather than unsupported; a GGUF that fails this and whose
  original `tokenizer.json` loads was damaged by its converter. Before this
  check the same file loaded and then failed on some prompts instead.
- `prompt length N bytes exceeds the maximum allowed prompt length of M bytes`
  is a 400 from `/tokenize`, `/v1/completions` or `/v1/chat/completions`. The
  server refuses a prompt it could never serve BEFORE tokenizing it, and it
  never truncates one. There is no option to raise the limit, because it is
  derived rather than configured: it is `max_model_len` multiplied by the
  longest token in the loaded vocabulary, so any prompt above it needs more
  than `max_model_len` tokens and would be refused after tokenizing anyway.
  Send a shorter prompt, or load a checkpoint with a longer context.

- A video render writes `<output_dir>/phase-log.json` beside its frames, and
  `unaccounted_seconds` there is time the render spent inside no named phase.
  Read `gaps` to find out WHERE: it holds one interval before each named phase
  and one after the last, each naming the two phases it lies between, and they
  add to `unaccounted_seconds` exactly. The largest entry is the region worth
  naming next. Subtract `instrument_seconds` first — that is what the
  instrument itself spent on its own phase boundaries, and on a short render it
  can be about half the residue. Every phase record carries its own
  `instrument_seconds` too, which is what that phase paid for the boundaries of
  its sub-phases. The C ABI hands back the same file's path through
  `vllm_video_last_phase_log`.

## Find a focused guide

[Task guides](guides/README.md) cover workflows that apply to more than one
model family, including offload, compatibility, and backend-specific use.

## Find a model recipe

[Model recipes](models/README.md) route you to commands, required weights,
component-specific runtime settings, and known limits for each model family.

## Checkpoint registry

This table identifies the checkpoints used by the model recipes. A model page
lists other published arms when they have not been used as a gated checkpoint.

A SHA-256 is required for a quantized artifact, because a repository id alone is
not a pin: checkpoints get re-quantized in place under an unchanged name. The
three LTX-2.5 rows added by [#1702](https://github.com/mudler/vllm.cpp/issues/1702)
carry one whether or not they are quantized, and the two that are not say so
beside it. That is deliberate rather than tidy: the value was **derived by
hashing the local bytes**, and it agreed with the etag the download recorded, so
each of those rows states a fact that was checked instead of one that was
reported. An etag nothing re-derived is not a pin here — an unauthenticated
HuggingFace tree API has returned a fabricated content hash for a gated
repository in this project's history.

<!-- checkpoint-registry:begin -->
| Model or component | File | Size | Repository and revision | Quantized SHA-256 | Supported arms | Refused arms or missing part |
|---|---|---|---|---|---|---|
| DSpark for Qwen3.8-27B | `model.safetensors` | 2,718,576,122 bytes | `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153be924f17ce4bf62726954eeaa4a73e854` | n/a (non-quantized) | Qwen3 DSpark routing | Token-exact decode gate is pending |
| Qwen3.8-27B EXL3 3.5bpw (the #2495 benchmark target; RUNS and GENERATES on CUDA) | `model-0000{1,2}-of-00002.safetensors` | 15,338,408,461 bytes total (index `total_size` 15,338,106,948 plus the two safetensors headers); 2426 tensors | `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` @ `19441ac874c4018295da848e250f23511361cda4` | `7b77214fe58ff15fed0b4af55e3cd92f38842b8711886d68954e8071ff8270c6` (shard 1), `411c83bb1070b27f3d670fc93e38dca0f17eb66429f64b5706901b12613188b2` (shard 2) — both **recomputed on `dgx:gpu0`** and matching the download host | `Qwen3_5ForConditionalGeneration`, 64 layers as 48 `linear_attention` + 16 `full_attention`. **409 trellis modules, every one codebook 2 (`mul1`), zero `mcg`**: bits 3 x137 (`mlp.{gate,up,down}_proj` of 46 layers), bits 4 x270 (all 48 GDN `linear_attn` x3, 55 more `mlp.*`, the 16 `self_attn.*`), bits 5 x1, bits 6 x1 (`lm_head`). It also ships its own quantized MTP head, 39 tensors under `mtp.*`, uniformly bits 4. GENERATES on GB10 greedy at `rc=0`; 16.7 tok/s target-only and 48.7 tok/s with its DFlash2 draft ([`qwen38-27b-exl3-gb10`](benchmarks/qwen38-27b-exl3-gb10.md)) | Its vision tower (`language_model_only: false`, depth 27) is not exercised. `quantization_config` says `out_scales: "always"` but the artifact ships **no scale tensors** — every quantized module is exactly `{trellis, suh, svh, mul1}` and the setting is folded into `svh`. The `m <= 8` GEMV had an arm for **none** of its 409 modules until 2026-09-02 -- its only arm was `(3,1)` and this artifact has zero `(3,1)` tensors. `(3,2)` is now instantiated, which is the 137 bits-3 modules, and `(4,2)` too, which is the other 270 ([#2570](https://github.com/mudler/vllm.cpp/issues/2570), [`QUANT-EXL3-PERF`](../.agents/specs/quant-exl3-perf.md)); 407 of the 409 now have an arm, and the 5- and 6-bit pair have none upstream either. **Whether the arm is TAKEN is a separate question from whether it exists**: `Exl3GemvSelectConfig` is upstream's envelope verbatim and declines on Blackwell below an occupancy threshold -- `narrow_coresident >= 544` at k 5120 n 17408 and `>= 160` at k 17408 n 5120, and 32, 160, 192, 320, 384 and 544 across the six bits-4 shapes -- and that HAS now been measured on GB10: `SM_COUNT=48 MAX_THREADS_PER_SM=1536` ceilings `narrow_coresident` at 144, so 34 of the 409 modules are admitted and the decode A/B is a null (17.16 vs 17.14 tok/s mean over three interleaved rounds, 1.3% spread), with `nsys` showing zero `exl3_gemv_kernel` launches at the default. The wide config does not escape it here, because its band needs `size_k <= 4096` and the smallest 4-bit `k` in this artifact is 5120. No throughput change is claimed here |
| Qwen3.8-27B ModelOpt NVFP4, the SGLang comparator's own target | `model-0000{1,2,3}-of-00003.safetensors` | 9,965,652,544 + 9,985,757,064 + 1,970,287,672 bytes; 2194 tensors | `RadixArk/Qwen3.8-27B-NVFP4` @ `554ebba9b5f1b79dc11246341960360e6ef05ef4` | `fbcdb5ba1cdda462b5f38592d071e772c4d398afea61a0aa9188b32d1a239a79` (shard 1), `db6146a5464fb0a891181b93c81593f0ca65c602eb14120a1c2b1b09bca11f85` (shard 2) and `d3cfb92742e30c8b46564665791dbe0a86ed64cfc02b1275081530793c0c9581` (shard 3), each **computed from the local bytes** and each matching the publisher's LFS object hash. Shard 2 was fetched as five disjoint HTTP ranges and reassembled after a single stream stalled, so its hash is the check that the reassembly is the file rather than a plausible-looking one. Leg B of `/workspace/nvfp4-sota/job1.sh` recomputes all three again on the device | ModelOpt `MIXED_PRECISION`: **208 per-tensor static FP8** (`self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`) and **193 NVFP4 `group_size` 16** (every `mlp.{gate,up,down}_proj` plus `lm_head`), all 401 carrying an `input_scale`. Both whole-checkpoint refusal gates answer empty on this `config.json` and its 2194 shipped tensor names, and shard 1 agrees with the committed `r0b0tlab` fixture on the dtype and shape of all 970 tensors they share. **The device load is leg E of that job and has not run**, so no arm here is claimed to execute | Its 193 NVFP4 modules are declared `NVFP4`, which is W4A4, and this build takes the **W4A16 weight-only** arm against them, because routing is by tensor name and `VT_MODELOPT_W4A4` defaults to `0`. Since `QUANT-QWEN38-27B-NVFP4-ARM` W7 the load SAYS so: one stderr notice naming both algorithms, the 193 modules, the 193 `input_scale` divisors it drops and the knob. The divergence is unpaid, not silent ([#2760](https://github.com/mudler/vllm.cpp/issues/2760)). The declared `kv_cache_scheme` FP8 is unread and bf16 KV is used, which is what the comparator also runs. The 15 bf16 `mtp.*` tensors are present and MTP execution is owed. The vision tower (`language_model_only: false`, depth 27) is not exercised |
| Qwen3.8-27B DFlash2 drafter, ModelOpt NVFP4 (the SGLang comparator's own drafter) | `model.safetensors` | 1,550,153,248 bytes; 186 tensors | `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal` @ `bd7a934213c47a9e7ef69eef36bb3325f47fd1f1` | `2228b9b22e93a88d84556419c879448ab6c490ae65c4c0b166f4962190ddbf26`, **computed from the local bytes** and matching the publisher's LFS object hash | **None.** Its `dflash_config` is byte-identical to the EXL3 DFlash2 drafter this tree does load (`block_size` 8, taps `[5,19,33,47,61]`, `selector_rank` 256, `selector_top_k` 16, `mask_token_id` 248070), and its five `layer_types` are all `sliding_attention` at `sliding_window` 2048 | **REFUSED, and not by name.** 35 of its Linears are ModelOpt NVFP4 `group_size` 16, and the DFlash2 draft loader has only a BF16 arm and an EXL3 arm; a draft's `quantization_config` is never read. Its own `exclude_modules` leaves `fc` unquantized, so the EXL3 probe on `fc` answers false, the BF16 reader is chosen, and the load dies inside layer 0 with `qwen3_dflash: expected BF16 for layers.0.self_attn.q_proj.weight`, a message that never says NVFP4 ([#2758](https://github.com/mudler/vllm.cpp/issues/2758)). Predicted from source; the verbatim device string is leg F of that job |
| Qwen3.8-27B DFlash2 draft, EXL3 5.0bpw (the drafter for the row above) | `model.safetensors` | 1,470,916,078 bytes; 189 tensors | `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` @ `4f0436269bca761b071f05319e8e04a87cc633f9` | `6b2e3afc694a343b7f3f0edfe5925e460762fc9ede4699165b577ca0733c8e56` | `DFlash2DraftModel`, 5 sliding-attention layers, `block_size 8`, taps at target layers `[5,19,33,47,61]`. **36 trellis modules, all bits 5 codebook 2** — uniform, NOT the "module-adaptive" the upstream README claims. It owns no `lm_head` and no `embed_tokens`: both are SHARED from the target, and the head is consumed **packed** (2,542,796,800 B saved on device). Drafts on GB10 at k=7 with output token-identical to the target-only arm | The candidate selector is **not** quantized, contrary to that README — `candidate_selector.hidden_projection` is dense F16 and its two `[248320, 256]` codebooks are raw BF16, together ~17% of the file. Its `kernel_projection` and selector weights are **F16** where the surrounding norms are BF16, admitted by name only. The shipped **paged** draft route faults on run 2 in one process ([#2274](https://github.com/mudler/vllm.cpp/issues/2274)); `VT_DFLASH_PAGED=0` is the measured arm |
| Nemotron-3.5-Lightning-30B | `model-000{01..52}-of-00052.safetensors` | 21,583,809,748 bytes total | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` @ `29f2d1746d8f41e316523194b19018707749b1b1` | `672c8bda10fdec0256e0819e112d2aa3a936cc3e5d311a05fd3ff773ca9a44b9` (first shard) | Device bf16, GQA, NVFP4 experts, and the NVFP4 head (A2-Q2b, unmeasured); host FP8 Mamba2 | GGUF, MTP, and batched decode |
| MiniMax-H3 FL2VA | `MiniMax-H3-FL2VA-Q4_K_M.gguf` | 19,864,208,160 bytes | `realrebelai/MiniMax-H3_GGUFs` @ `daf03b4ca652cce16dfd4fcf91e79c52ffa5c1e7` | `5e8fa6e960d5fbd547390ceec63fcead275435d8f3bd2466a8a2cbd8c2e361e3` | Q4_K_M `t2va` and `fl2va`, verified end to end | `ref2va` requires the REF2VA partition |
| MiniMax-H3 REF2VA | `MiniMax-H3-REF2VA-Q4_K_M.gguf` | 19,864,208,064 bytes | `realrebelai/MiniMax-H3_GGUFs` @ `daf03b4ca652cce16dfd4fcf91e79c52ffa5c1e7` | `17925612821ea3037ffaf5f7f9789f5460e87025385bd45e9ec6c7d536684d56` | Q4_K_M `ref2va`, verified end to end | `t2va` and `fl2va` require the FL2VA partition |
| MiniMax-H3 encoder | `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | 14,576,977,888 bytes | `realrebelai/MiniMax-H3_GGUFs` @ `daf03b4ca652cce16dfd4fcf91e79c52ffa5c1e7` | `1bf75e038c5895b97b6ea16cc1e3d32076254b06ec3df10657650d86dc82279e` | Q4_K_M text and multimodal conditioning | No separate refused arm recorded |
| MiniMax-H3 pruned FL2VA | `minimax_h3_fl2va_pruned-Q8_0.gguf` | 21,437,786,208 bytes | `unsloth/MiniMax-H3-GGUF` @ `d629413c2e5b51b38c453668b75ca3b06ca92703` | `1c77759fd30e4b41dd4fb341d684518177f544428c6186fd9f5fd96f8ebf55d4` | Pruned Q8_0 loads and renders | Other pruned quant levels load but have not been rendered |
| MiniMax-H3 pruned REF2VA | `minimax_h3_ref2va_pruned-Q8_0.gguf` | 21,414,002,784 bytes | `unsloth/MiniMax-H3-GGUF` @ `d629413c2e5b51b38c453668b75ca3b06ca92703` | `60f8a47434ec9a925f0aea41d9e0db9cb78ebc46791b7488d621dbd6905e5d89` | Pruned Q8_0 loads and renders | Other pruned quant levels load but have not been rendered |
| MiniMax-H3 video VAE | `FL2VA/video_vae/source/model.safetensors` | 10,415,548,320 bytes | `MiniMaxAI/MiniMax-H3` @ `42ed227ee7df40d41602854ae760620d6eb651fe` | n/a (non-quantized) | Official video decode for the five-file recipe | No quantized arm is recorded |
| MiniMax-H3 audio VAE | `FL2VA/audio_vae/model.safetensors` | 605,429,308 bytes | `MiniMaxAI/MiniMax-H3` @ `42ed227ee7df40d41602854ae760620d6eb651fe` | n/a (non-quantized) | Official audio decode for the five-file recipe | No quantized arm is recorded |
| MiniMax-H3 tokenizer | `FL2VA/tokenizer/tokenizer.json` | 7,032,403 bytes | `MiniMaxAI/MiniMax-H3` @ `42ed227ee7df40d41602854ae760620d6eb651fe` | n/a (non-quantized) | Official tokenizer for the five-file recipe | No separate arm is recorded |
| MiniMax-Music3 | Diffusers checkpoint tree | about 28.5 GB resident | `MiniMaxAI/MiniMax-Music3` @ `fbdf52fbaaca799592917417eb05f1899f1255ec` | n/a (non-quantized) | bf16 language model, depth decoder, condition encoder; fp32 transformer and vocoder | Native `.pth` layout |
| MiniMax-Music3 depth decoder | `rvq_depth_decoder_q4_k.gguf` | 405,752,480 bytes | `audio-cpp/MiniMax-Music3-GGUF` @ `c36aaeed683f33b05796788e4204f4eeba8fa547` | `4c5d41b27418d9c1046345f649cb61d7cde0e3bbda4af7f7cb142df2c70cbdd0` | GGUF Q4_K depth decoder | Other GGUF components and third-party lineages |
| LTX-2.5 full DiT | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` | `792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584` (non-quantized; hashed anyway, see the note above this table — derived 2026-08-27 from the bytes the upstream oracle render loaded) | Full bf16 DiT; declare `--checkpoint-class full`. GATED: this is the DiT #1854's absolute comparison rendered on (`rc` job `4b0666ee`, `PASS`), and the sha256 beside it was recomputed inside that lease from the locally staged copy as well as from the oracle's own bytes | A mismatched or missing required class is refused |
| LTX-2.5 distilled DiT | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42,018,190,584 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` | n/a (non-quantized) | Distilled bf16 DiT; declare `--checkpoint-class distilled` | A mismatched or missing required class is refused |
| LTX-2.5 distilled NVFP4 DiT | `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18,721,432,024 bytes | `Lightricks/LTX-2.5` @ `8a4ff96f581e72bedc1b44367581c49d544a05f1` | `f9c4c2ae9a6aa8f732eb02a1c4c3b34888caad3dd35bb65deaf3b5043cda78fa` | Distilled NVFP4 DiT, 7876 tensors | The same path at `6c7e5e57...` is a different artefact, and the next section gives both value sets |
| LTX-2.5 distilled LoRA | `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 8,899,889,568 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` | n/a (non-quantized) | REQUIRED by every non-distilled two-stage recipe — `ti2vid_two_stage`, `keyframe_interpolation`, `a2vid_two_stage`, `res2s_two_stage` and `dfr` — and applied to both stages on the last two; rank and alpha 450; version 2.5.0 | A load that omits it on those five arms is refused by name; distinct from the 327,322,640-byte IC-LoRA |
| LTX-2.5 video VAE | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | 1,452,269,922 bytes | `Lightricks/LTX-2.5` @ `8a4ff96f581e72bedc1b44367581c49d544a05f1` | `685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d` (non-quantized; hashed anyway, see the note above this table) | The `--video-vae` argument of every render; the CONV VAE, which is what the shipped recipes pass | The DiffVAE sibling `ltx-2.5-video-vae-bf16.safetensors` is refused by name rather than silently downgraded |
| LTX-2.5 audio VAE | `vae/ltx-2.5-audio-vae-bf16.safetensors` | 364,866,540 bytes | `Lightricks/LTX-2.5` @ `8a4ff96f581e72bedc1b44367581c49d544a05f1` | `c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5` (non-quantized; hashed anyway, see the note above this table) | The `--audio-vae` argument of every render | No quantized arm is recorded |
| LTX-2.5 Gemma-4 12B text encoder, bf16 | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | 26,263,858,182 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` (gated) | `ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1` (non-quantized; hashed anyway, and derived three independent times — the download's `x-linked-etag`, a CIFS read, and the worker's local disk during the render) | The **upstream oracle's** text tower, and the only one it accepts: `tools/oracle/ltx2_oracle.py` and #1864's reference render. This project's loader now reads it too: its two caption projections are stored BF16 [4096, 188160] and [2048, 188160] with no scale tensor in the file, and until [#2140](https://github.com/mudler/vllm.cpp/issues/2140) the loader doubled that already logical width to 376320 and refused. That refusal was MEASURED and LOCALISED on 2026-08-27 (`rc` job `001c36e9`): the 12 B tower itself loaded in bf16 in 34.815 s, and only the two caption projections refused. Measured on these bytes, not inferred | Unlike the torchao row below, this file DOES carry a `__metadata__` block, so `--encoder-config` is not required beside it. **This tower is now the GATED one for #1854.** `rc` job `4b0666ee-248c-45fc-9de6-372b6d0c1fab` on `dgx:gpu0` rendered the oracle's exact request (320x192, 25 frames, 8 steps, seed 42) against it and the absolute comparison returned `PASS` / `NO_WORSE_THAN_ORACLE_ON_BLOCKINESS`, so the arm-matched comparison that row promised has been taken. Other LTX-2.5 renders in this tree still take the NVFP4 torchao tower in the row below, which is a DIFFERENT arm and not interchangeable with this one. Upstream reads no torchao tensor at pin `fd4ded7f`, so the two are not interchangeable in either direction |
| LTX-2.5 Gemma-4 12B text encoder | `text_encoders/gemma4-12b-with-proj-nvfp4-torchao.safetensors` | 7,423,624,178 bytes | `vonkaiser/LTX-2.5-FP8-NVFP4` @ `5a40ba9ab209a90ddb7943d1e3d374c51cfd3256` | `12132b7157925332d2b21de9fc6f507c14f4f0cbc7081484d1968ebf8a19b4bf` | The `--encoder` argument of every render, NVFP4 torchao | This file carries NO `__metadata__` block, so `--encoder-config` is REQUIRED beside it and the loader refuses by name without it (`ltx2_text_encoder.cpp`) |
| LTX-2.5 prompt-adherence scorer (an INSTRUMENT, not an oracle) | `pytorch_model.bin` plus `config.json`, `preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json` | 598,641,023 bytes for the weights, 602,356,502 bytes for all eight | `openai/clip-vit-base-patch16` @ `57c216476eefef5ab752ec549e440a49ae4ae5f3` | `ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0` for `pytorch_model.bin`; all eight measured digests are in `tests/parity/goldens/ltx25_adherence/scorer-pin.json`, which the tool checks before it reads a pixel | What `scripts/ltx25-render-compare.py --adherence-model` scores prompt adherence with (#1854 sub-question 1, #2295). It is upstream's own choice twice over: vLLM registers the family as `CLIPEmbeddingModel` (`registry.py:251` at `5559679229`) and vLLM-Omni's accuracy suite scores video prompt-faithfulness with this exact checkpoint (`tests/e2e/accuracy/helpers.py:497`). **THREE COSTS A READER MUST NOT DISCOVER LATER: the HuggingFace repository DECLARES NO LICENCE** (`cardData.license` is null; the `openai/CLIP` GitHub repository is MIT, but the weights repository asserts nothing), **it ships a PICKLE and no safetensors**, and **its text context is 77 positions**, so a longer prompt is REFUSED and never truncated. The #1864 reference request is 17 CLIP tokens and fits; #1854's own 70-word golden-retriever prompt needs at least 83 and does not, so this scorer cannot answer the example that issue uses to define the problem. Its tower is 224x224, so a 320x192 frame reaches it as a resized centre crop | **REFUSED: any prompt over 77 CLIP text positions**, and it is refused rather than truncated, so #1854's own 70-word golden-retriever example (at least 83 positions) cannot be scored by this instrument at all. **REFUSED: a checkpoint whose sha256 does not match the pin, or a pin whose digest is null.** **MEASURED, and our render FAILS: our own engine's frames at the reference's request** were scored on 2026-09-01 from the 25 that `rc` job `93a60151-7d4d-4718-842c-ef724208be0e` retained on the share. The true prompt still ranks first over all six decoys, by +0.3370 on 15 of 25 frames, so the render depicts the asked-for scene. Its mean CLIP score is 35.2719 against a bound of 36.0087 taken from the reference's own frames, a margin of -0.7368, so it depicts it less well than upstream's and #1854's first sub-question stays open on a measured shortfall. **That reading is ONE render, n = 1**: the lease retained only its first render's frames, so the run-to-run stability of our own adherence score is UNMEASURED, and a reading that moved by 0.74 between runs would make this verdict a coin toss rather than a finding. The numbers are in `.agents/specs/ltx25-prompt-adherence.md` under `## Outcome`. **WHY it falls short is also measured now, and the answer is not the one the record predicted**: `scripts/ltx25-adherence-detail-loss.py` loads this same checkpoint through this same identity check ([#2513](https://github.com/mudler/vllm.cpp/issues/2513)) and REFUTES the smoothness hypothesis -- measured border-free our render carries 1.4031x the reference's absolute high-band power at 1.0373x its mid-band power, so there is no rolloff to call smoothness. The intervention that settles the direction is on the REFERENCE's frames: blurring them until their sharpness is a fifth of ours costs only 1.9687 CLIP points against an observed gap of 2.7305, while still ranking the true prompt first on 23 of 25 frames, so no achievable smoothing reproduces our gap. **A "blurring OUR frames RAISES their score by 1.9131" figure was published here and is WITHDRAWN**: from sigma 0.50 upward a decoy outranks the true prompt on our blurred frames, so those arms are not adherence readings at all, and the only readable row of that sweep moves our score by -0.0029. That row also first published the band figure as "77.79% MORE" from a windowed whole-frame spectrum and WITHDREW it along with the separable-upsampler cause it had suggested; its `## CORRECTION` carries the four-convention table and `## CORRECTION 2` the arm-validity finding. **MISSING: `CLIPEmbeddingModel` itself** -- the ported upstream test runs the HuggingFace half only, because `VllmRunner` needs an installed vLLM and a GPU lease, so vLLM's own runner has never been loaded here. **NOT MEASURED: temporal adherence** -- frames are scored independently, exactly as vLLM-Omni's own middle-frame scorer does. Downloaded under authority recorded in `.agents/developer-preferences.md` (developer, 2026-08-31); loaded through `transformers` at f32, which is upstream's own dtype for this path (`test_clip.py:75`, `dtype="float"`) rather than a widening of ours; needs no GPU |
| Qwen3.8-27B GGUF language model | `Qwen3.8-27B-Q4_K_M.gguf` | 17,106,775,008 bytes | `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | Q4_K_M text model loads through `--model` and decodes on CPU | **The token gate against llama.cpp `b10451` FAILED** on 2026-08-23: tokenizer exact 6/6, generation divergent 5/6 ([evidence](bench-evidence/qwen38-27b-q4km-token-gate-20260823.md), #821). GGUF multimodal forward is missing |
| Qwen3.8-27B GGUF Unsloth-Dynamic language model | `Qwen3.8-27B-UD-Q4_K_M.gguf` | 16,464,440,224 bytes | `unsloth/Qwen3.8-27B-GGUF` @ `4ca720788d1e01f1bff70c033e0d0028fd02e502` | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` | The reader accepts every one of its 866 tensors. This is the artifact `IQ3_S` (ggml id 21) was added for: 4 of those tensors carry it and `GgufFile::Open` refused the whole file over them until #2510, and 4 blocks of `blk.11.ffn_gate.weight` from it are the oracle goldens the decoder is gated on. **The SHA-256 was derived by hashing the local bytes**, not read off a tree API. | The `IQ3_S` keep-quant `vec_dot` is owed, so those 4 tensors EXPAND to bf16 on the GEMM arm of every device (146.13 MiB of blocks against 680.00 MiB of bf16, 3.4 % of the file); no token gate and no throughput number is claimed on this file — #2497 owns the `gfx1151` quant-matched decode number and #2510 owns the `vec_dot` |
| Qwen3.8-27B GGUF projector | `mmproj-BF16.gguf` | 931,146,432 bytes | `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` | BF16 `clip` projector loads and validates through `--mmproj` | No request path runs the loaded projector |
| Qwen3.6-27B GGUF, the LIMB-3 VEHICLE (a DENOMINATOR, not a capability) | `Qwen3.6-27B-Q4_K_M.gguf` | 16,817,244,384 bytes; 851 tensors | `unsloth/Qwen3.6-27B-GGUF` @ `82d411acf4a06cfb8d9b073a5211bf410bfc29bf` | `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`, hashed THREE times and equal each time: by the fetch script as it landed, independently off the staged share afterwards, and a third time on the `strix:gpu0` worker's own copy after the transfer out of `/workspace`. It also equals the digest the forge advertises for that revision. Its vision tower `mmproj-BF16.gguf`, same revision, is 931,146,304 bytes, sha256 `05353347512982ee62317b9d8c89372bc815f4b4043580e7ef3ad411ec1a1cd3`; the tokenizer and config come from `Qwen/Qwen3.6-27B` @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | ARM: Q4_K_M `qwen35`, `block_count` 64, histogram F32 449 / Q4_K 289 / Q5_K 48 / Q6_K 65 — **pure k-quant, with no non-k-quant weight tier at all**, so it exercises the same `DotQ4K`, `KQuantGemmK`, `QuantizeQ8KK` and `MatmulBTQuantKernelRocm` path the Qwen3.8-27B Q4_K_M ROCm arm runs. It LOADS AND GENERATES on `strix:gpu0`: 3 legs on the shipped default with no knobs, 2 clean and token-identical to each other, 1 `BOARD_FAULT` (GPU Hang, rc 139), zero `[vt reference-tier]` hits. Fetched under authority recorded in `.agents/developer-preferences.md` (developer, 2026-09-04), scoped to ONE vehicle meeting the six conditions [#2864](https://github.com/mudler/vllm.cpp/issues/2864) pre-registered, all six of which it meets ([`docs/bench-evidence/limb3-vehicle-pin-20260904.md`](bench-evidence/limb3-vehicle-pin-20260904.md)) | **THIS CHECKPOINT DID NOT DELIVER LIMB 3, AND THE REASON IS THE ORACLE RATHER THAN THIS FILE.** `STRICT_LIMB3 = NO` ([#2884](https://github.com/mudler/vllm.cpp/issues/2884), [`docs/bench-evidence/limb3-strict-gate-20260904.md`](bench-evidence/limb3-strict-gate-20260904.md)): the pinned vLLM is **not deterministic on this vehicle**, so there is no single denominator to be token-exact against. Its eager and compiled configurations each reproduce themselves exactly and **disagree with each other on 2 of 6 prompts**, and each also disagrees with its own one-pass prefill argmax (eager 3 steps, compiled 2). The strict free-running counts are 3/6 against eager and 3/6 against compiled — **recorded, and NOT usable as a gate result**, and not even the same three prompts. Whether any divergence is an exact tie is **NOT ESTABLISHED**: the run captured `prompt_logprobs=1`, so no runner-up margin exists. **REFUSED AS A CONCLUSION: picking whichever vLLM configuration agrees with us**, which `score_strict.py` forbids in code by short-circuiting on the oracle's self-consistency before it compares our tokens. No throughput, latency or memory figure was taken from this artifact and none may be quoted; `STRIX_ARM_SPEED_RATIFIED_BY` stays unset |
| Qwen3.8-27B mixed FP8 and NVFP4 | `model.safetensors` | 22,568,192,096 bytes | `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | `c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05` | NVFP4 modules load | FP8 modules and quantized KV cache are refused |
| Qwen3.8-27B MTP drafter | `model_mtp.safetensors` | 849,400,392 bytes | `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | n/a (non-quantized) | BF16 MTP artifact is present | MTP execution is owed |
| Qwen3.8-27B ModelOpt NVFP4 shard 1 of 4 | `model-00001-of-00004.safetensors` | 9,965,644,108 bytes | `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a22990e82c54c1d48ee77c491b87825680` | Locally computed hash is owed; the bytes are not mirrored here, and the four shards are verified semantically instead (header parse plus data end equal to the published size); #821 | 208 per-tensor static FP8 and 193 W4A16_NVFP4 modules load | The declared FP8 KV cache is unread; #1593 |
| Qwen3.8-27B ModelOpt NVFP4 shard 2 of 4 | `model-00002-of-00004.safetensors` | 9,985,743,924 bytes | `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a22990e82c54c1d48ee77c491b87825680` | Locally computed hash is owed; #821 | Same arms as shard 1 | The declared FP8 KV cache is unread; #1593 |
| Qwen3.8-27B ModelOpt NVFP4 shard 3 of 4 | `model-00003-of-00004.safetensors` | 1,120,886,516 bytes | `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a22990e82c54c1d48ee77c491b87825680` | Locally computed hash is owed; #821 | Same arms as shard 1 | The declared FP8 KV cache is unread; #1593 |
| Qwen3.8-27B ModelOpt MTP drafter | `model-00004-of-00004.safetensors` | 849,400,592 bytes | `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a22990e82c54c1d48ee77c491b87825680` | Locally computed hash is owed; #821 | Fifteen BF16 MTP tensors are present and unquantized | MTP execution is owed |
| Qwen3.8-2.4T-A95B | `UD-Q1_0` ten-file GGUF split | about 370 GiB | `unsloth/Qwen3.8-2.4T-A95B-GGUF` @ `567d3e6ac26c5474b18311e619c04350fb9a5556` | `b7770552b2ac24e7334c917bc92e90e218e87cfe29484db65e62e8ef2a60334d` (shard 1); `2765517f833c736338d3ab34354e1c10eb8d79e62325f998285b435e5cf03dcd` (shard 2) | CPU expert streaming from disk | CUDA refuses a checkpoint that exceeds device capacity |
| Llama-3.2-1B-Instruct EXL3 3.0bpw (the first EXL3 checkpoint that GENERATES) | `model.safetensors` | 1,089,087,416 bytes | `turboderp/Llama-3.2-1B-Instruct-exl3` @ `f8f438c290680b15622270eff03bef23a458b1cf` (revision `3.0bpw` -- this repo publishes ONE BRANCH PER BIT WIDTH and `main` carries no weights at all, so a bare repo id resolves to nothing) | `3c0341e9c7c4c16a86a499de1dff4f6d7de9855541d669f3b0e214d72b54c2fc` | LOADS and GENERATES end to end through `vllm-cli` on `--device cpu`: `The capital of France is` -> ` Paris. Paris is known for its famous landmarks such as the Eiffel Tower` (greedy, 16 tokens, 2026-08-28). Native exllamav3 layout, no `.rank{r}` slicing; the body is 3-bit and `lm_head` is SIX-bit, resolved per tensor | Codebook **0** (the original QTIP 3INST), because the artifact ships no `mcg` marker and `LinearEXL3` derives the codebook from tensor PRESENCE. **It now RUNS ON CUDA**: the device arm instantiates `(3,0)`, `(3,1)` and `(6,0)`, so this checkpoint's 3-bit body and 6-bit head both reach the GPU. **It now RUNS ON ROCm too**, on `strix:gpu0` (`gfx1151`, RDNA3.5 APU, ROCm 7.2.4, 2026-09-02): the same file completes greedy generation with ZERO CPU reference-tier operations at 8.27 tok/s warm, against 0.83 tok/s for the same tree with the two ROCm registrations disabled, and the two arms emit the IDENTICAL continuation ` Paris. Paris is known for its famous` ([#2433](https://github.com/mudler/vllm.cpp/issues/2433)). That is an EXL3-vs-EXL3 A/B and NOT a ratio against a bf16 target: the BF16 control hung the GPU in the same lease ([#2511](https://github.com/mudler/vllm.cpp/issues/2511)), so this row records no BF16 denominator and no AMD clock attribution. gfx1151 is an APU with unified memory, so the figure generalizes to no discrete Radeon -- and on a discrete board the reference tier is off by design, which means this arm is what makes EXL3 RUN there at all rather than what makes it fast. It has NO GEMV fast path at `m == 1` and takes the regular shape table, which is upstream's behaviour too — its envelope refuses `bits != 4 && cb == 0` and its instantiation list omits `(3,0)`. **No speed is claimed.** The device figure ranged 2.1-5.0 tok/s across jobs on one binary and box, and an interleaved comparison against the BF16 twin of the same model read 2.33 vs 2.24 tok/s — indistinguishable, so the remaining ~50x to the memory floor is per-step engine overhead and not this scheme ([#2233](https://github.com/mudler/vllm.cpp/issues/2233)). q/k/v and gate/up run as separate GEMMs rather than one merged operand |
| DeepSeek-V4-Flash EXL3 trellis shard 1 of 172 | `exl3-layer-000-tp4-rank0.safetensors` | 515,850,920 bytes | `0xSero/deepseek-v4-flash-0731-spark` @ `22f28d32b9b29b4352eaa380ff8c2c170b2847ab` | `2ed7ae798a794019810b027fe2609e2cf4ad78d70b49c47b2970d03a0a7aaadf` | The rank-sliced EXL3 routed-expert tower LOADS (TP4 coalesced to TP1) and its experts EXECUTE through `vt::Exl3Gemm` on a CPU queue | The CUDA arm compiles for `sm_121a` and its numeric gates PASSED on GB10 on 2026-08-28 (`had_r_128` byte-identical, `exl3_gemm` `rel_rms 5.538e-4`, GEMV tier 3c `5.160e-4`); the FUSED MoE device arm still cannot run, because it needs a device-resident tower, so the routed experts execute on a CPU queue. That run decoded ZERO tensors of THIS artifact -- it found no readable shard -- so nothing here is a claim about these weights on a device. A SYNTHETIC rank-sliced checkpoint now loads and emits logits end to end; THIS artifact still does not, because its DSA compressor and indexer tensors are stored at twice the width the host forward indexes (`compressor.wgate` `[2*head_dim, H]`) and the loader refuses them BY NAME, and because its tokenizer is not read ([#1924](https://github.com/mudler/vllm.cpp/issues/1924)) |
| DeepSeek-V4-Flash EXL3 carried tower shard 1 of 5 | `carried-001.safetensors` | 4,288,630,252 bytes | `0xSero/deepseek-v4-flash-0731-spark` @ `22f28d32b9b29b4352eaa380ff8c2c170b2847ab` | `3b67ae29f1e75c2ecadfcafd3b0eecec640b06fd60b832f77e6bd3c2a8c85ccf` | The un-requantized `deepseek_v4_fp8` attention, router, shared-expert, compressor and embedding tensors, MATERIALIZED at load into the host-float tower the forward composes with — block-wise FP8 (`F8_E4M3` + `F8_E8M0` over 128x128 blocks) decoded to f32, BF16 norms and embeddings widened, I64 `tid2eid` narrowed to int32 | The DSA compressor and indexer tensors of this artifact are `2 * head_dim` / `2 * index_head_dim` wide and the loader refuses them by name (41 of its 43 layers carry a compressor); the 3,985 `mtp.*` NVFP4 draft tensors are skipped and counted, never silently dropped |
| GLM-5.3-Flash FP8 source | `model-000{01..62}-of-00062.safetensors` | 328,326,771,576 bytes total (305.78 GiB) | `zai-org/GLM-5.3-Flash` @ `main`, read 2026-08-26 | Owed: no byte of payload has been fetched, so no local hash exists to state, and an unauthenticated tree hash is not a pin here | Declared source of `scripts/convert-glm5-next-gguf.py`. Only the safetensors HEADERS were read, by HTTP RANGE over all 62 shards: 76,108 tensors, `F8_E4M3` block-quantized at `weight_block_size: [128, 128]` with `weight_scale_inv` companions, plus BF16 and F32 scales | **Nothing has been converted.** The download needs explicit developer authority and a box with room for 305.78 GiB of source and ~100.35 GiB of output at once; owed as O7 on [#2011](https://github.com/mudler/vllm.cpp/issues/2011). The revision is a branch name and not a commit, which is NOT a pin: it is what was read, and W7b re-reads and records the commit when it stages the bytes |
| GLM-5.3-Flash GGUF | `GLM-5.3-Flash-UD-Q2_K_XL-0000{1..4}-of-00004.gguf` | 108,720,071,427 bytes total (101.2535 GiB) across four shards; 1412 tensors | `unsloth/GLM-5.3-Flash-GGUF` @ `d425e572fb9686125831f476129e51cea34bc5b4`, path `UD-Q2_K_XL`, staged 2026-08-28 | Owed for this row: the shards are staged and were sha256-verified when they were fetched, but **W5c consumed only the four GGUF HEADERS** and states no hash of its own. W7b ([#2225](https://github.com/mudler/vllm.cpp/issues/2225)) records the per-shard sha256 alongside the load it measures | **LOADS on `--device cpu`, and the engine's multi-KV guard no longer refuses above the model's forward** ([#2348](https://github.com/mudler/vllm.cpp/issues/2348)). **A MATERIALIZED LOAD EXISTS** -- driven at this artifact on `dgx:gpu0` 2026-08-30, all four shards load and the engine sizes its caches in under 26 minutes wall ([#2343](https://github.com/mudler/vllm.cpp/issues/2343)). At that change the first step threw at the `multi_kv` guard above the model's own hook; W5b-2c ([#2348](https://github.com/mudler/vllm.cpp/issues/2348)) writes the consuming forward that guard was waiting for and it no longer fires for this model. **THIS ARTIFACT GENERATES COHERENT TEXT, and peak RSS is MEASURED** as of [#2241](https://github.com/mudler/vllm.cpp/issues/2241). On `dgx:gpu0` 2026-08-30, in the SHIPPED configuration with no diagnostic env set, `vllm-cli --device cpu --max-tokens 2` at the prompt `The capital of France is` emits ` Paris.` at `rc=0`, and `VmHWM` peaks at 104,792,300 kB = 99.94 GiB. Two instrumented `thor:gpu0` runs the same day supply the bisect: four tokens read ` Paris. Paris is`, the prefill top-5 is ` Paris` (16.427), ` one`, ` located`, ` known`, ` a` at margin 1.279, and none of 180 per-layer readings over four steps carries a NaN. The first attempt emitted token id 0 eight times, because the loader repacked this file's 346 q8_0 tensors into the i8mm interleave that the host bridge reads as plain blocks (spec `## Owed` O30). **No speed number is claimed, and the earlier ones are void** -- they were taken from an all-NaN forward. The GB10 arm is the one measured above. The GGUF arm of `load_weights` resolves all 1383 backbone tensors of this file (W5c, [#2242](https://github.com/mudler/vllm.cpp/issues/2242)); `blk.45`, the multi-token-prediction block, is read, counted and DROPPED, as the transformers reference does. `ModelRegistry::Forward` dispatches to the model as of W5b-2b ([#2337](https://github.com/mudler/vllm.cpp/issues/2337)), which bridges ONE decoder layer at a time out of the block-resident tower and decodes only the 8 of 288 experts a token selects — a float tower is 426.72 GiB against ~119.63 GiB usable. **A MATERIALIZED LOAD NOW EXISTS**: driven at this artifact on `dgx:gpu0` 2026-08-30, all four shards load and the engine sizes its caches in under 26 minutes wall. **NO TOKEN WAS GENERATED** — the first step throws at the `multi_kv` guard above the model's own hook ([#2343](https://github.com/mudler/vllm.cpp/issues/2343), [#2068](https://github.com/mudler/vllm.cpp/issues/2068)) — and **peak RSS and speed are still unmeasured**, because the staging run did not sample them. The vision tower (a separate `mmproj-BF16.gguf`) and the safetensors arm still refuse by name, as does a multi-request step; a non-CPU queue is admitted as of W9c-3a ([#2464](https://github.com/mudler/vllm.cpp/issues/2464)) for the routed-expert GEMM alone, and a device that is neither CPU nor CUDA is refused by name; **the KV-cache spec does not**, as of W5 ([#2223](https://github.com/mudler/vllm.cpp/issues/2223)), which publishes its three groups through the production factory hook | **The earlier row here said `none exists`, and that was true when it was written (2026-08-26) and is not now.** "UD-Q2_K_XL" names a TARGET AVERAGE and not a format: the census over all 1412 tensors is F32 638, Q8_0 346, Q5_K 181, Q6_K 117, IQ2_XS 82, IQ3_XXS 41, IQ4_XS 3, Q2_K 2, Q4_K 1, Q3_K 1 — **two** Q2_K tensors in a file named Q2_K. It fits `dgx:gpu0` only because IQ2_XS and IQ4_XS keep their blocks ([#2247](https://github.com/mudler/vllm.cpp/issues/2247)); both now have a CUDA keep-quant kernel too ([#2260](https://github.com/mudler/vllm.cpp/issues/2260)), so the expert GEMM no longer drains the stream to the host and the fused seam no longer throws. W9c-3a ([#2464](https://github.com/mudler/vllm.cpp/issues/2464)) then built a device arm for this artifact's routed-expert GEMM and MEASURED it end to end, where it **SEGFAULTED**: both `--device cuda` legs on `dgx:gpu0` died with rc=139 emitting no token, reproducibly (spec O46). The split is therefore OPT-IN and defaults OFF, so `--device cuda` refuses exactly as it did before. **Use `--device cpu`** -- measured on that artifact it emits ` Paris.` at rc=0, 1176 s wall of which 169 s is generation. Every OTHER primitive of this model is still a host reference on an interposed CPU queue (spec O43), so what `--device cuda` reaches is one arm of eleven and not a device arm. **A materialized load NOW exists and a token still does not** — `dgx:gpu0` 2026-08-30 ([#2343](https://github.com/mudler/vllm.cpp/issues/2343)): all four shards load and the engine sizes its caches, then the first step throws at the `multi_kv` guard above the model's own hook. **Peak RSS and speed remain unmeasured** |
| GLM-5.3-Flash config | `config.json` | 69,416 bytes | `zai-org/GLM-5.3-Flash` @ `main`, read 2026-08-27 | sha256 `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f` | The ONLY byte of this checkpoint any change on this row has consumed. Checked in verbatim as `tests/vllm/models/fixtures/glm5_next/config.json` and used as W1's gate fixture, so the config layer is gated against what the checkpoint says rather than against what a port's author believed it says | **Arms refused by name:** the SAFETENSORS one, which is what this row is, because every published safetensors artifact of this model exceeds every device this project owns. `Glm5NextForConditionalGeneration` is REGISTERED, its config RESOLVES, and the GGUF arm both loads and forwards ([#2067](https://github.com/mudler/vllm.cpp/issues/2067), [#2242](https://github.com/mudler/vllm.cpp/issues/2242), [#2337](https://github.com/mudler/vllm.cpp/issues/2337)). The revision is a branch name and not a commit, which is NOT a pin for the WEIGHTS; for this one file the sha256 above is the pin |
| GLM-5.3 GGUF (`glm-dsa`) all six shards | `GLM-5.3-UD-IQ1_S-0000{1..6}-of-00006.gguf` | 9,428,677 B (shard 1, metadata only, 0 tensors) and 49,968,868,928 B (shard 2); the six shards total 216,715,365,893 B = 201.83 GiB across 1809 tensors | `unsloth/GLM-5.3-GGUF` @ `346b3591c7f28d1a23716f97a065ecf12ec14771`, path `UD-IQ1_S`, staged 2026-08-30, completed and verified 2026-08-31 | shard 1 `ff3adab0853dfb00bdf3889ec3f5556196f56b65783115720d57767bbd760dd9`; shard 2 `659d04cf4fc0b6026944f34c0b590a635803bff06c1775361e28490db7b168f8`; shard 3 `433302bac0e2d54da64c7c2f28509fa1b235aeccdf5b215a8a446ebaad1b5b27`; shard 4 `d0a6f19452d5b5cd498e1eb8fbe856e00aed7da1f80c27c095301eabe81e9bc1`; shard 5 `2ea1537ffab40fa8b8584a8647ec10fbaa6199dfed45e4019b822da2b319db37`; shard 6 `42a76ef04ffc5e321e1240f4e572b6fa6fc3315da5bea22fb598d7460db210fe`. **All six are complete and each was hashed TWICE** — once by the fetch script as it landed and once independently off the same share afterwards — and the two readings agree. **The DERIVED metadata shard has a hash of its own:** `scripts/glm-dsa-write-indexer-types.py` run against the staged shard 1 with `zai-org/GLM-5.3`'s own `config.json` produces a 9,428,810-byte file, 64 keys becoming 65, 21 `full` of 78, sha256 `b3e9838651a5c279533c98390ab4bc03cf1d8c176d5be0754180f07d9ed85c01`, reproduced identically by three independent runs. **That is a DERIVED artifact and must never be quoted as `unsloth/GLM-5.3-GGUF`'s shard 1** | **THIS ARTIFACT GENERATES THROUGH THE EXPERT-STREAMING LANE: `The capital of France is` -> ` Paris`.** On `dgx:gpu0` (GB10, 20 cores, 119 GB, compute capability 12.1) under an `rc` lease, 2026-08-31, a build with `-DVLLM_CPP_CUDA_ARCHITECTURES=121a -DVLLM_CPP_FLASH_ATTN=ON` and CUTLASS 4.5.0: `VT_MOE_EXPERT_STREAM=1 VT_MOE_EXPERT_STREAM_SLOTS=4096 vllm-cli --model <derived shard 1> --device cuda --prompt "The capital of France is" --max-tokens 1 --temperature 0` returns `rc=0`, `prompt_tokens=5 completion_tokens=1`, and seven bytes of stdout: a space, `Paris`, a newline. Wall 1154 s for the process, `generate` 852.330 s, `VmHWM` 60,512,268 kB = 57.71 GiB against 119.631 GiB of device and 201.83 GiB of artifact. **The lane's own counters are the streaming evidence, and they include one number that must travel with them:** `[expert-stream] ON slots=4096 slot_bytes=6684672 resident=25.50 GiB`, then `steps=1 hits=0 misses=6399 evictions=0 fills=4096 bytes=13939408896 exhausted=2303 advised=0`. 4096 slices were paged out of the file into slots and 12.98 GiB moved through them with zero evictions, and the 187.312 GiB of towers were never materialized — but the step needed 6399 distinct slices, so **2303 of them (36%) were read in place out of the mapping instead of streamed**. That is a PREFILL working set exceeding any slot budget by construction (spec R2, O34), it is counted rather than silent, and no figure here may be quoted as a fully-streamed step. **No speed number is claimed:** one token, a CIFS-backed artifact, and 2303 in-place fallbacks in the measurement. `--device cpu` on the same box and artifact also emits ` Paris` (`rc=0`, `generate` 950.249 s, `VmHWM` 44.46 GiB), and **that arm does NOT stream at all** — a CPU queue builds no slot lane, so every routed-expert slice is read in place. On `thor:gpu0` (sm_110a) the CUDA arm cannot reach a token: MLA prefill on this family IS FlashAttention, the vendored FA2 covers `8.0,8.6,8.7,8.9,12.0a,12.1a`, and sm_110a is outside it. Also gated on a complete synthetic model of the same shape: `test_glm_moe_dsa_gguf_load.cpp` 5 cases / 228 assertions, `test_glm_moe_dsa_forward.cpp` 7 / 5258, `test_glm_moe_dsa_schedule.cpp` 12 / 533, and the real file's census from its headers (`test_glm_moe_dsa_gguf_census.cpp` 3 / 3831): 1809 tensors, 228 expert towers at 187.312 GiB, 1581 resident at 14.511 GiB, largest per-expert slice 6,684,672 B. What the forward still refuses BY NAME is a step in which any request RESUMES while its selection PRUNES — that needs the indexer KV side cache `KV-DSV4-MULTICACHE` owns (spec O4, #1925/#2323), so a FIRST token on a fresh prompt is reachable and a SECOND is not — and sparse prefill (spec O6) is still W6's. No speed axis has a denominator (spec O10) | **THIS FILE CANNOT BE FED AS PUBLISHED**, and that is a property of the file rather than of the port: its 64 metadata keys carry neither `glm-dsa.attention.indexer.types` nor `index_topk_freq`/`index_skip_topk_offset`, so it states its per-layer indexer schedule nowhere, and it broadcasts `indexer.*` onto all 79 blocks while the checkpoint ships them on 22. The loader refuses it by name rather than substituting llama.cpp's hardcoded table (spec D3). **The repair is one command and it rewrites the 9.4 MB metadata shard only:** `scripts/glm-dsa-write-indexer-types.py --shard <shard 1> --from-config <zai-org/GLM-5.3 config.json> --out <dir>/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf`, with the five payload shards hard-linked beside the output, then `--model` that directory's shard 1. It transcribes the schedule from the model author's own `config.json` and derives nothing; the result is a DERIVED artifact with its own sha256 and is not `unsloth/GLM-5.3-GGUF`. **Build requirements this model does not degrade past:** `--device cuda` (the expert-streaming lane is not built on a CPU queue, and the towers would then be read in place out of a 201.83 GiB mmap), and a build with the vendored FlashAttention-2, which needs CUTLASS headers and an arch in `8.0,8.6,8.7,8.9,12.0a,12.1a` — MLA prefill IS FlashAttention here and has no fallback below it. **On ROCm `gfx1151` (`strix:gpu0`, Radeon 8060S) this artifact NOW GENERATES TEXT, as of [#2572](https://github.com/mudler/vllm.cpp/pull/2572): `The capital of France is` -> ` Paris, which is`.** It loaded but emitted nothing between [#2562](https://github.com/mudler/vllm.cpp/pull/2562) and that change. The route is `VT_CPU_MOE=1 vllm-cli --model <derived shard 1> --device auto` -- `auto` because no `--device` value names ROCm ([#2505](https://github.com/mudler/vllm.cpp/issues/2505)), and `cpu_moe` because `--fit`'s default placement leaves 22 layers on a device whose keep-quant set cannot hold their IQ1_S towers ([#2565](https://github.com/mudler/vllm.cpp/issues/2565)). All 1809 tensors resolve, all 228 routed-expert towers stay compressed, 11.620 GiB is paged in (at 11.5 MiB/s off the CIFS share the artifact lives on, which is a property of the share), and the engine auto-fits `max_model_len` to 8192 against 256 blocks of 32 tokens. **TOKENS NOW COME OUT, and what serves them must travel with them.** The first forward used to throw in the MLA block, because `vt::OpRegistered(kFusedNormRope, ...)` was false on ROCm and cannot see the reference tier, so the split A-projection path was taken and refused this checkpoint's block-quantized `kv_a_proj_with_mqa` ([#2564](https://github.com/mudler/vllm.cpp/issues/2564)). Registering a native ROCm `kFusedNormRope` makes that predicate true, and the run completes: measured 2026-09-02, `rc` job `6b35b8d3-be7f-4d71-abf4-0f0bd72bb643`, `VT_CPU_MOE=1 VT_OP_PROVIDER_STATS=1 vllm-cli --model <derived shard 1> --device auto --prompt "The capital of France is" --max-tokens 4 --temperature 0` returns `rc=0`, `prompt_tokens=5 completion_tokens=4 finish_reason=length`, and prints ` Paris, which is`. **FIVE ops ran on the portable CPU reference tier in that run** -- `ConcatAndCacheMla`, `ConcatMlaNopeRope`, `MlaPrefillAttention`, `BatchedMatmul` and `MlaDecodeAttention` -- and the `kFusedNormRope` this change adds is NOT one of them (`op=114 device=5 selected=vt-native`). **NO SPEED NUMBER IS ADMISSIBLE from this run and none is offered:** `docs/ROCM.md` disqualifies any performance result with a non-zero reference-tier hit count, and this run has five. The 3516.719 s the harness printed for four tokens is recorded here only as the cost of a host-tier MLA arm, never as a throughput result. The streamed-expert lane is NOT what serves the towers here and cannot be: `pageableMemoryAccess` is 0 on this board, so `host_memory_is_device_addressable()` is false ([#2515](https://github.com/mudler/vllm.cpp/issues/2515)). No speed number is admissible from this board for this model. **Arms refused by name:** the SAFETENSORS one, permanently (spec D1 — 703.74 GiB across 141 shards, no streaming loader, no MoE block-fp8 rung), and `UD-IQ1_M`, which refuses at file open because `IQ1_M` (ggml id 29) has no reader traits (spec O3) |
| GLM-5.3 config | `config.json` | 29,464 bytes | `zai-org/GLM-5.3` @ `935644c05e76fc198714f4cca449fd8b970ff6d7` | Committed verbatim in-tree as `tests/vllm/models/glm_moe_dsa_config_glm53.inc`, so the config layer is gated against what the checkpoint says rather than against what a port's author believed it says | It is the ONLY authoritative source of the 78-entry `indexer_types` list — 21 `full`, at layers {0,1,2} and every fourth from 6 to 74 — which three independent derivations agree on bit for bit (the list itself, vLLM's rule at `deepseek_v2.py:1097-1101`, and llama.cpp's `GLM_5_2_DEFAULT_INDEXER_TYPES`) | The GGUF above does not carry this list, which is why it cannot be fed as published |
| Qwen3.5-0.8B (Tenstorrent P150 arm) | `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 bytes | `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`, authorized 2026-08-23 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` (non-quantized; hashed anyway from the local bytes the gates and the eager profile consumed) | bf16 on the Tenstorrent P150: the sacred greedy pair, both ambient legs, and the #1715/#2107 profile legs all ran from this snapshot | **Arms refused by name:** GGUF k-quant arms on TT — no TT kernels exist for them, refused at load; Qwen3.8-27B on TT — no arm fits the P150 (bf16 53.8 GB), refused at load |
| dots3-note bf16 language tower | `model-000{01..131}-of-00131.safetensors` | 561,371,869,568 bytes total (522.82 GiB), of which the MoE is 545,823,175,680 | `dots-studio/dots3-note-prev` @ `1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b` | Owed: **no tensor byte has been fetched**, so no local hash exists to state, and an unauthenticated tree hash is not a pin here | The bf16 text tower this port loads: 46 backbone layers, both MLA geometries, and since W5 the 45 MoE layers — the ungrouped noaux_tc router at 256/8 plus one shared expert at `moe_intermediate_size * n_shared_experts` = 1536. Everything except `mlp.gate.e_score_correction_bias` is BF16; that one is F32, on both sides | **Nothing has ever loaded these bytes.** The tower alone is 522.82 GiB against a 122 GiB ceiling on the largest host this project reaches (spec §6.2), so the arm is representable and unfeedable, and the e2e gate is an OPEN GAP by construction. GGUF k-quants are refused by name (W9). The 19-tensor nextn tail is a NAMED W10 deferral rather than a refusal since #2176 |
| dots3-note vision tower | `model-vision.safetensors` | 13,742,557,056 bytes | `dots-studio/dots3-note-prev` @ `1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b` | Owed, as above | none | **W6a ported the DENSE half and W6b ([#2613](https://github.com/mudler/vllm.cpp/issues/2613)) the PYRAMID one, so this checkpoint's tower LOADS.** All 2195 of its `vision_encoder.*` tensors are read: 235 dense (`patch_embed`, blocks 0-24, the `patch_merger` adapter) plus the 1960 the pyramid adds (17 routed blocks x 8, and 608 routed experts x 3). The 17 F32 tensors in this otherwise all-BF16 file are exactly the `mlp.router_bias` buffers, one per routed block, which is upstream's own `dtype=torch.float32` (`vision.py:152-155` @ `9035151d6`) and is asserted in both directions at load. An image request against this tower is SERVED, and since W6c ([#2537](https://github.com/mudler/vllm.cpp/issues/2537)) it no longer has to be a multiple of 28 on each side: `PilResizeBicubicRgb` ports Pillow's own `Image.Resampling.BICUBIC` — support scaled by `max(1, in/out)` on a downscale, weights normalized per output pixel, 22-bit fixed point across two passes over a uint8 intermediate — so the resample upstream always performs now happens here too instead of being refused. What still refuses BY NAME is the blockwise-FP8 sibling below (W9); no bf16 dots3-note `vision_config` published so far selects the `use_bias` arm ([#2616](https://github.com/mudler/vllm.cpp/issues/2616)) or the softmax / top-k-below-2 router arms ([#2615](https://github.com/mudler/vllm.cpp/issues/2615)), which are the only bf16 configurations left owed |
| dots3-note audio tower | `model-audio.safetensors` | 1,772,399,360 bytes | `dots-studio/dots3-note-prev` @ `1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b` | Owed, as above | `added_tokens.json` sha256 `1aa71a4e0dbab80a72fd925389fd6c9cc52d1cb9da5dee8282784c15c6fa789b`; `tokenizer.json` sha256 `7f4e21a1d9fa472439f70201b4849977da5ec11e73df5a36552ab5ee99af554b` | **W7a ([#2703](https://github.com/mudler/vllm.cpp/issues/2703)) puts this tower on a SERVED request.** All 430 `audio_encoder.*` tensors are read and every one is BF16 — not one F32 in the file. An OpenAI `input_audio` part reaches the 32-layer `dots` speech encoder through `ApiServer::handle_chat_completions` on the default configuration. The three audio markers are resolved BY STRING from this checkpoint's own tokenizer, which carries them as special added tokens `<|audio_comp_start|>` 151718, `<|audio_comp_end|>` 151719, `<|audio_comp_pad|>` 151720 — note `pad == start + 2`, not `start + 1`. **W7b ([#2797](https://github.com/mudler/vllm.cpp/issues/2797)) removed the `chunk_seconds` ceiling**: a recording of any length is sliced into `chunk_seconds` segments, each padded to `chunk_samples` and log-melled on its own, run through the tower at ITS OWN valid sample length, and concatenated IN ORDER (`nvidia/audio.py:193-234` @ `9035151d6`). On this checkpoint `chunk_samples` is 960000 = 750 token strides, so the tower's per-segment row sum and the prompt side's single `ceil(total/stride)` agree for every waveform; a checkpoint whose `chunk_samples` is NOT a whole number of `token_stride`s is refused per request past one chunk, at the chat seam. **W7c-1 ([#2813](https://github.com/mudler/vllm.cpp/issues/2813)) lifted the MONO restriction**: a multi-channel PCM16 WAV already at 16 kHz is served, reduced to mono by the per-sample mean over its channels, which is upstream's own reduction (`load_audio(..., mono=True)` -> `np.mean`, `vllm/multimodal/media/audio.py:207-208`, `:220` @ `9035151d6`; and `ChannelReduction.MEAN` with `AudioSpec.target_channels = 1`, which dots3-note selects at `vllm/models/dots3_note/common/processor.py:523-525`). **W7c-2 ([#2828](https://github.com/mudler/vllm.cpp/issues/2828)) lifted the 16 kHz restriction**: a PCM16 WAV at any sampling rate is served, resampled to this checkpoint's `audio_config.sampling_rate` before the front end. This is the row's one RECORDED DIVERGENCE. Upstream's default resampler is `pyav`/libswresample, which is not bit-identical to itself across CPU dispatch on one binary and one input, so a bit-exact gate against it is impossible in principle; what is ported is `resample_audio_scipy` (`vllm/multimodal/audio.py:232-250` @ `9035151d6`), an arm of upstream's own `AudioResampler` switch that vLLM ships in production for phi4mm, and the gate is a consistency gate against `scipy.signal.resample_poly` with committed goldens. **Refused BY NAME**: a non-positive sample rate, and a reduced polyphase ratio whose `max(up, down)` exceeds 100000 — a deliberate divergence, because the rate is named by the request's own WAV header and the anti-alias filter is `20 * max(up, down) + 1` taps, while every ordinary rate reduces far below the bound (44100 -> 441, 48000 -> 3, 22050 -> 441, 8000 -> 2); any container but RIFF/WAVE PCM16 — `mp3`, `flac`, `ogg` need a demuxer this tree does not vendor, and that is owned by the shared codec brick [#2814](https://github.com/mudler/vllm.cpp/issues/2814) rather than by this row; and the unshipped `audio_config` arms (`use_causal`, `use_conv1d_stem`, `use_latent_input`, `merge_factor != 1`, a non-`dots` `encoder_type`), none of which this checkpoint selects. The blockwise-FP8 sibling below is still W9 |
| dots3-note blockwise-FP8 sibling | `model-000{01..131}-of-00131.safetensors` plus the two tower files | 298,673,280,504 bytes total (278.16 GiB) across 133 safetensors, read 2026-08-28 | `dots-studio/dots3-note-prev-fp8` @ `7c14222e22423d6df6848eb0d1c5c3a88a00311a` | Owed: only `config.json` and `model.safetensors.index.json` were read | none | **Refused BY NAME at the forward, naming W9.** Its `quantization_config` is `{"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}` and its index (73,029 entries) ships a `weight_scale_inv` beside every projection — at the routed experts' `[1536, 5120]` that scale is `[12, 40]`. This port's bf16 loaders read a per-tensor or per-output-ROW `<name>_scale` and nothing else, so without the named refusal the load would fail with a bare "tensor not found". It does not fit either: 278.16 GiB against the same 122 GiB ceiling |
| Qwen3.8-Flash-Next GGUF | `Qwen3.8-Flash-Next-UD-IQ1_S-0000{1..3}-of-00003.gguf` | 72,546,461,344 bytes total (67.564 GiB) across three shards (10,946,624 + 49,990,818,368 + 22,544,696,352); 1224 tensors | `unsloth/Qwen3.8-Flash-Next-GGUF` @ `8bdc666649440e9bdc97e16f3f75782c98478ff5`, path `UD-IQ1_S` | `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd` (shard 1); `3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6` (shard 2); `0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a` (shard 3). Shard 1's digest was recomputed TWICE for this row -- on the development box and again INSIDE the `thor` lease against the bytes the server actually opened. Shards 2 and 3 carry the digests recorded in [the ladder-arm evidence file](bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md), which recomputed all three on the staged copy on 29 August 2026; **this wave did not re-derive those two**, because the hash was killed mid-run for reading the same CIFS share as the load being measured | **LOADS on `--device cpu`, and the server LISTENS -- it produces NO TOKEN.** Measured on `thor:gpu0` 2026-08-30 (`rc` job `0f188dd1`, [evidence](bench-evidence/qwen4exp-released-checkpoint-serve-20260830.md)): all three shards load through `LoadedEngine::FromModelDir`, the engine sizes all three published cache groups, the tokenizer and the 9993-character chat template come out of the GGUF's own metadata, and `examples/server` answers on `/health`. **Load wall time 4446 s (74.1 min); peak RSS `VmHWM` 69.206 GiB against a 67.564 GiB artifact.** Residency is keep-quant: anonymous memory moved 4 -> 11 GiB across a load whose n-gram table alone would have added 95.368 GiB there, so all nine encodings in the file (F32, Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ1_S, IQ4_NL, BF16) keep their blocks. `POST /v1/completions` then returns **500** and zero tokens | **THE FORWARD REFUSED THIS ARTIFACT BY NAME ON THAT RUN, AND W5p REMOVED THE REFUSAL**: `vt: qwen4_exp_gated_residual: input_mix_weight_down must be float (f32/bf16 for outputs)`. The file stores all **194** hyper-connection mix weights (`blk.N.hc_{attn,ffn}_{down,up}.weight` and `output_hc_{down,up}.weight`) as **Q8_0**; our loader correctly keeps them quantized (`qwen4_exp_weights.cpp` -> `LoadMatmul`), and `vt::Qwen4ExpGatedResidual` accepted only float, while every arm of the synthetic fixture wrote those same names as ggml type 0 (F32) -- so every prior wave gated the float case only and none could see this. Since W5p the three PROJECTION operands (`mix_down`, `mix_up`, `block_inject`) accept a block-quantized `[N,K]` weight and route through `vt::MatmulBT`/`kMatmulBTQuant`, mirroring llama.cpp, which merged this architecture on 2026-08-27 (`6c84c7d5d`, first tag `b10660`) and declares all six of them `GGML_OP_MUL_MAT`; the ELEMENTWISE `hc_*_norm` gamma is still refused by name, which is llama.cpp's own split. `FixtureOpts::hc_mix_q8_0` is the fixture arm that was missing. **W5q RE-RAN THIS ARTIFACT ON 2026-08-31** ([evidence](bench-evidence/qwen4exp-released-checkpoint-serve-20260831.md)): staged to worker-local disk it loads in **61 s** rather than 4446 s, `VmHWM` 73.935 GiB, the prefill and eight decode steps complete with nothing thrown, and `POST /v1/completions` returns **200** with 8 tokens. **Every token was id 0 (`!`) and two different prompts returned a byte-identical answer.** **W5s RE-RAN IT ON 2026-08-31 ON `origin/main` `52f7ccbfc`, WHICH CARRIES W5r, AND THE TOKENS ARE REAL** ([evidence](bench-evidence/qwen4exp-released-checkpoint-tokens-20260831.md)): `"The capital of France is"` -> `" Paris. Given this fact, what is"` and `"Water boils at"` -> `" 100°C at sea level"`, eight distinct token ids none of them 0, loaded in 60 s from the staged copy at `VmHWM` 73.93 GiB with system `used` flat at 11 GiB. **The cause of W5q's degeneracy was the dropped repack marker W5r fixed**: on this aarch64 i8mm box `kMatmulBTQuant` had been reading `block_q8_0x4` buffers as flat `q8_0`, putting a NaN in layer 0 that collapsed to an all-zero logit row, and `argmax` over a row with no maximum returns index 0. `VT_CPU_QUANT_REPACK=0` now gives byte-identical output to the default. **WHAT RUNS IS EXACTLY THIS AND NO MORE: `--device cpu`, ONE SEQUENCE AT A TIME, the UD-IQ1_S arm.** It is **NOT a token gate** — no oracle decoded these prompts, and there is no speed number. ISSUE OWED (this account is suspended for GitHub **API** writes -- `gh issue create` returns `HTTP 403: Sorry. Your account was suspended`, while `git push` over SSH succeeds, which is how this row reached `main`); scoped under `## Owed` in [the spec](../.agents/specs/qwen4-exp-flash-next.md). **Also refused or absent:** the other six published quants (UD-IQ1_M, UD-Q2_K_XL, UD-IQ3_XXS, UD-Q3_K_XL, UD-IQ4_XS, UD-Q4_K_XL) are staged but **none has been run**; every safetensors artifact (~360 GB bf16, ~180 GB FP8, ~128 GB NVFP4) exceeds the 122.80 GiB of the largest box in this fleet; the n-gram table stays HOST-side on every arm, because `DeviceQuantGatherSupported` is true for `kCPU` alone and moving it would expand it from 26.822 GiB to 95.368 GiB ([#2083](https://github.com/mudler/vllm.cpp/issues/2083)) -- the clause here previously said that "any non-CPU device refuses by name ahead of tensor I/O", which the CUDA run below falsifies; **`--device cuda` NOW SERVES THIS ARTIFACT, FLUENTLY BUT NOT TOKEN-EXACTLY** (the sentence here previously read that `ModelRegistry::Forward` is all-or-nothing and "no `qwen4_exp` step reaches a CUDA queue", which stopped being true once every op on the path had a device arm): on `thor:gpu0` `sm_110` the same binary answers `The capital of France is` with `11751 13 15767 411 1928 11 628 567` against the CPU control's `11751 13 15767 411 2029 11 1092 369` -- five of eight ids, both continuations grammatical English. **BOTH ARMS NOW MIRROR vLLM: since [#2612](https://github.com/mudler/vllm.cpp/issues/2612) the CPU GDN prefill runs vLLM's chunked decomposition too, by default, so the CPU/CUDA distinction this cell used to draw is GONE.** **BOTH SEQUENCES ABOVE WERE RE-MEASURED UNDER THE NEW DEFAULT ON 2026-09-04 AND NEITHER MOVED** ([evidence](bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904.md), [#2858](https://github.com/mudler/vllm.cpp/issues/2858)): same artifact, same prompt, production configuration, `11751 13 15767 411 2029 11 1092 369` on `--device cpu` and `11751 13 15767 411 1928 11 628 567` on `--device cuda`, still five of eight. This cell previously said those CPU ids were "expected to move and have NOT been re-measured"; that expectation is falsified, and `VT_GDN_CHUNKED=0` now emits the same eight ids as the default rather than a different set. **The chunked arm is not inert** -- on the same binary it moves decoder layer 0's Gated DeltaNet block output by `3.702e-04` and brings the CPU arm to `1.772e-05` of CUDA where the sequential arm sat `3.525e-04` away, a 19.9x reduction -- it simply flips none of the eight argmaxes, which is what the row's spec says to expect from an argmax over near-ties. With the Gated DeltaNet source closed, the whole-model gap is now dominated by the undiagnosed MoE residue and `out` improves only 1.23x. The sequential arm is still the more ACCURATE one -- it lands `1.15e-08` from the exact answer where vLLM's own chunked kernel lands `2.29e-04` ([decomposition](bench-evidence/gdn-chunked-decomposition-20260902.md)) -- and it is retained for exactly that, but accuracy and faithfulness are different things and the ids vLLM would emit are the chunked arm's. The two arms are not a defect apart: the first tensor that differs is decoder layer 0's Gated DeltaNet block output, from a bit-identical input, because the CUDA arm ran vLLM's chunked prefill decomposition and the CPU arm an exact sequential recurrence ([evidence](bench-evidence/qwen4exp-cuda-prefill-divergence-20260902.md), [#2547](https://github.com/mudler/vllm.cpp/issues/2547)) -- that divergence SOURCE is closed by [#2612](https://github.com/mudler/vllm.cpp/issues/2612), which put both arms on the chunked decomposition, though the row makes no token-agreement claim and none should be read into it; `num_reqs > 1` is refused by name; MTP is absent (**zero** `nextn`/`mtp` tensors of 1224 against 31 in the safetensors repo, [#1993](https://github.com/mudler/vllm.cpp/issues/1993)); and the file is TEXT-ONLY (no `v.blk.*`), so the multimodal arm has no artifact |
<!-- checkpoint-registry:end -->

### Convert a GLM-5.3-Flash checkpoint to GGUF

`zai-org/GLM-5.3-Flash` (`Glm5NextForConditionalGeneration` / `glm5_next`)
publishes no arm that fits any device this project reaches, and no upstream tool
can make one: llama.cpp has no `glm5_next` at our pin `b10451` or at its
`master`, and `gguf-py`'s `Q2_K` has a dequantizer and **no** quantizer. So the
converter ships here.

```sh
# Read the headers and print the plan, without writing a byte.
scripts/convert-glm5-next-gguf.py --src /path/to/GLM-5.3-Flash --arm q2_k --dry-run

# Write the arm.
scripts/convert-glm5-next-gguf.py --src /path/to/GLM-5.3-Flash \
    --dst GLM-5.3-Flash-Q2_K.gguf --arm q2_k
```

numpy is its only dependency. It streams shard by shard, so peak resident memory
is one tensor rather than one shard, but the output is written in one pass and
needs its full size free on the destination.

| `--arm` | what the experts get | everything else | weights on the real model |
|---|---|---|---|
| `q2_k` | Q2_K | Q6_K | **100.35 GiB** — the only arm that fits ~119.63 GiB |
| `q6_k` | Q6_K | Q6_K | 239.89 GiB |
| `q8_0` | Q8_0 | Q8_0 | 310.67 GiB |
| `bf16`, `f16`, `f32` | passthrough | passthrough | 584.67 GiB at bf16 |

Routed and shared experts are 97% of this model, so the arm name is the expert
type and the remaining 3% rides at a finer one almost for free. Figures are the
converter's own per-tensor plan over the real topology (1719 tensors, 313.89B
parameters after the layer-45 MTP block is dropped), not bits-per-weight times a
parameter count.

**Refused by name, each with the missing part.** Every i-quant — `iq1_s`,
`iq2_xxs`, `iq2_s`, `iq3_xxs`, `iq4_xs`, `iq1_xxxs` — needs an importance
matrix, an importance matrix needs a forward pass over the model, and the
smallest published artifact is 181.32 GiB, so the dependency is circular on this
fleet. `q3_k`, `q4_k` and `q5_k` are refused because those encoders are not
ported: only Q2_K, Q6_K and Q8_0 are ported from `ggml/src/ggml-quants.c` at the
pinned llama.cpp `b10451` and gated byte-for-byte against it.
`--keep-mtp` is refused because nothing here reads an MTP tail.

**A `glm5next` file LOADS, and the model's forward now CONSUMES the engine's
paged KV cache set instead of being refused above.** W1 ([#2067](https://github.com/mudler/vllm.cpp/issues/2067)) gave
`glm5next` its `general.architecture` dispatch row and registered
`Glm5NextForConditionalGeneration`, so passing such a file to a `.gguf` entry
point reads its metadata, cross-checks its per-layer schedule against its tensor
inventory, and validates its config — through the same parser a `config.json`
descends through. W5c ([#2242](https://github.com/mudler/vllm.cpp/issues/2242))
landed the weight tower, so the load COMPLETES: every tensor group the
architecture declares is mapped, and a missing tensor, a disagreeing shape or an
`ssm_a` that is not the negated exponential the container writes is refused BY
NAME. W5b-2b ([#2337](https://github.com/mudler/vllm.cpp/issues/2337)) landed
the forward hook, which `ModelRegistry::Forward` dispatches to: the tower stays
block-resident exactly as loaded, one decoder layer at a time is decoded to host
f32 and dropped, and each token's 8 routed experts of 288 are decoded on demand.
A float tower would be 426.72 GiB against ~119.63 GiB usable on the largest box
this project reaches, so this is the only shape that fits and not an
optimization.

**THE ENGINE PATH NOW REACHES THAT HOOK.** Until W5b-2c
([#2348](https://github.com/mudler/vllm.cpp/issues/2348)) it did not, and that
was MEASURED rather than assumed: driven at the staged 101.2535 GiB artifact on
`dgx:gpu0` on 2026-08-30 ([#2343](https://github.com/mudler/vllm.cpp/issues/2343)),
all four shards load and the engine sizes its caches -- `max_model_len` auto-fits
from 1048576 to 8192 against 256 blocks of 32 tokens, and `max_num_seqs` drops
from 32 to 1 because one 4,390,912-byte GDN state fills a unified page -- and the
FIRST step then threw:

```text
vt: model forward: 22 KV cache(s) from 2 published group(s) reached this forward,
first 'model.layers.3.self_attn.attn', with block tables gathered for 3 of 3
published group(s), and no registered forward consumes a cache set keyed by
layer name.
```

Those are the bytes emitted at the SHA that run was measured on. The guard now
names the arriving architecture and leaves ownership of the consuming forward to
that architecture's own row ([#2353](https://github.com/mudler/vllm.cpp/issues/2353)),
and it now reports the paged/recurrent split, because a total that counted only
attention caches omitted the 34 recurrent states while the block-table
denominator still counted their group (`ENG-MULTIKV-BYNAME`). A run today reads
`architecture 'Glm5NextForConditionalGeneration' reached this forward with 56 KV
cache(s) (22 paged, 34 recurrent) ...` and says in the message what the paragraph
below had to say in prose. The transcript is kept as measured rather than
rewritten, because it is dated evidence and not a specimen of current output.

That is the `multi_kv` guard at the TOP of `ModelRegistry::Forward`, landed by
KV-DSV4-MULTICACHE W3 ([#2068](https://github.com/mudler/vllm.cpp/issues/2068)),
and it fires for ANY model publishing a multi-cache topology BEFORE dispatch to
that model's hook. W5b-2c is the consuming forward it was waiting for:
`ForwardGlm5NextForConditionalGeneration` maps the engine's layer-name-keyed
`MultiKvCacheIndex` onto the three groups `MakeGlm5NextKVCache` publishes -- the
11 DSA layers' MLA latent, the 34 KDA layers' recurrent state and the 11 indexer
side caches -- hydrates one layer state per layer out of the engine's paged
buffers, and writes each step's new rows back into them. The guard is NARROWED
and not removed: it still refuses any model whose forward does not declare that
it consumes a keyed cache set, DeepSeek-V4 included.

**WHAT IS STILL NOT MEASURED, and no number here should be read as one.** NO
TOKEN has been generated from this artifact at any change, and no peak RSS and
no throughput figure exists for it. A generation attempt at W5b-2c is OWED, and
its absence is a missing measurement rather than a passing one. No oracle
registers this architecture at any revision that can also RUN on a device this
project owns, so no end-to-end token gate for the 321.32B model exists or can
exist on this fleet; what is gated is a synthetic 4-layer miniature at
`hidden_size` 32, whose cached prefill-plus-continue agrees with its own
one-shot forward EXACTLY.

**Three things still refuse BY NAME, each naming what it owes.** A step carrying
more than one request, because this forward is single-sequence and concatenating
requests would attend across the boundary; a non-CPU queue, because every
primitive of this model is a host f32 reference and the device arm is owed; and
the VISION tower, which the `glm5next` container does not carry at all — the
published artifact ships it as a separate `mmproj-BF16.gguf`.

**What has NOT been measured, and no number here should be read as one.** No
token, no peak RSS and no throughput figure exists for this artifact. No oracle
registers this architecture at any revision that can also RUN on a device this
project owns, so no end-to-end token gate for the 321.32B model exists or can
exist on this fleet; what is gated is a synthetic miniature. The forward re-runs
the whole prefix each step and re-decodes every layer, which is a residency
decision rather than a speed one, and the speed axis is unopened.

**Use `--device cpu`, and the reason is no longer the quantization.** This
model's forward is a host f32 reference and refuses a non-CPU queue BY NAME,
before any GEMM runs, so `--device cuda` on this artifact is an error message
whatever the kernels underneath do; the device arm is owed by the model's own
row. What changed is the layer below it: the 82 IQ2_XS and 3 IQ4_XS tensors that
had no CUDA keep-quant kernel now have one
([#2260](https://github.com/mudler/vllm.cpp/issues/2260)), so the expert GEMM no
longer drains the stream to the host cores and the fused MoE seam no longer
throws. That drain was measured on GB10 to SEGFAULT rather than merely run
slowly, whenever the tensors came from the ordinary CUDA device allocator. That is a prerequisite for a CUDA arm of this model, not a CUDA arm.

**Our own converter has still never been run** against the real 305.78 GiB
checkpoint; that needs explicit developer authority for the download and a box
with room for source and output at once
([#2011](https://github.com/mudler/vllm.cpp/issues/2011),
[#2225](https://github.com/mudler/vllm.cpp/issues/2225)). What it writes moved
in [#2291](https://github.com/mudler/vllm.cpp/issues/2291) onto the container
convention the published artifact uses — `ssm_dt.bias`, a split
`attn_k_b`/`attn_v_b` with the k half transposed, and `ssm_a = -exp(A_log)` — so
one spelling is written and one is read.

### The distilled NVFP4 DiT was re-quantized under an unchanged name

`Lightricks/LTX-2.5` published two different files at
`diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`. The
registry row names the one this project read. Both value sets are recorded here,
because earlier evidence cites the superseded set
([#1723](https://github.com/mudler/vllm.cpp/issues/1723)).

| Field | Superseded value | Registry value |
|---|---|---|
| Revision | `6c7e5e573ac1667efc83407806fe9b0b93730e60` | `8a4ff96f581e72bedc1b44367581c49d544a05f1` |
| Size | 18,721,548,408 bytes | 18,721,432,024 bytes |
| Tensor count | 7877 | 7876 |
| Safetensors header | 1,287,600 bytes | 1,179,408 bytes |
| SHA-256 | Never obtained | `f9c4c2ae9a6aa8f732eb02a1c4c3b34888caad3dd35bb65deaf3b5043cda78fa` |
| Provenance | The HuggingFace `/api/models/Lightricks/LTX-2.5` tree listing, read on 17 August 2026, and an authenticated range request on 20 August 2026 | The bytes on the shared checkout, downloaded on 12 August 2026 and hashed on 24 August 2026 |

Neither value set is a transcription error. `main` moved between 12 August 2026
and 17 August 2026, and the file gained 116,384 bytes and one tensor across that
move. The superseded set describes the later artefact. Only its safetensors
header was ever read here, by the range request on 20 August 2026, and no run
loaded its tensors. The registry value set describes the earlier artefact, which
every LTX-2.5 NVFP4 measurement here used, so it is the set this table must
carry: the table identifies the checkpoints the recipes used.

The SHA-256 closes the gap that made the row unfixable before. It was derived by
hashing the local bytes at 66.1 MiB/s over 270 s, and it equals the etag the
`huggingface_hub` `.metadata` sidecar recorded for that download. An etag that
nothing re-derived is not a pin here, so the agreement is the evidence and the
etag alone was not.

The two halves of this pin do not have equal standing, and the difference is
recorded rather than smoothed over. The CONTENT half is re-derived: the SHA-256
comes from the bytes on the shared checkout and is reproducible by anyone who
holds them. The REVISION half is reported: `8a4ff96f...` is what
`huggingface_hub` wrote down about where it fetched the file on 12 August 2026,
and no fetch from that revision has been made since to confirm it. The support
for it is circumstantial and consistent. All six `Lightricks/LTX-2.5` sidecars
on the shared checkout record the same value, a download log records one
six-file wave that day, and the file modification times fall in two waves that
match. Treat the SHA-256 as the identifier of these bytes. Treat the revision as
the best available statement of where they came from.

Three LTX-2.5 rows still name `6c7e5e573ac1667efc83407806fe9b0b93730e60`: the
two bf16 DiTs and the distilled LoRA. No `.metadata` sidecar exists for any of
the three on the shared checkout, so nothing local confirms or contradicts their
revision, and each keeps the revision it was recorded with. Their sizes are not
in the same position. The full bf16 DiT and the distilled LoRA sit on the shared
checkout at exactly the recorded byte counts, so those two sizes are locally
confirmed. The distilled bf16 DiT is absent from the shared checkout, so its
42,018,190,584 bytes rest on the tree listing and a range request alone.

## Look up interface details

[Reference pages](reference/README.md) collect dense lookup material such as
build settings, environment variables, feature state, and release artifacts.

Native Windows release artifacts are not published yet. The Windows CPU and
Vulkan ZIP downloads do not exist until the `v0.0.3-pre.1` prerelease workflow
and post-publication audit succeed. See [Binary releases](RELEASES.md).
<!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished -->
