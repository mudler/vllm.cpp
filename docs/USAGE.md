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

**Not on the C ABI yet.** `vllm_model_params` carries no `kv_cache_dtype` field,
so a C-ABI caller reaches the fp8 cache only through a checkpoint that declares
it. Tracked by [#1593](https://github.com/mudler/vllm.cpp/issues/1593).

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
- On ROCm, GGUF mixture-of-experts checkpoints compute on the quantized
  expert blocks (Q8_0, Q4_K, Q5_K, Q6_K) instead of being dequantized to
  bf16 at load time.
- On ROCm, mixture-of-experts models run the shared-expert gate and both
  expert-combine steps on device. Before these ops were registered the
  engine refused with `no kernel for op` on that path.
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
| LTX-2.5 full DiT | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` | `792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584` (non-quantized; hashed anyway, see the note above this table — derived 2026-08-27 from the bytes the upstream oracle render loaded) | Full bf16 DiT; declare `--checkpoint-class full` | A mismatched or missing required class is refused |
| LTX-2.5 distilled DiT | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42,018,190,584 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` | n/a (non-quantized) | Distilled bf16 DiT; declare `--checkpoint-class distilled` | A mismatched or missing required class is refused |
| LTX-2.5 distilled NVFP4 DiT | `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18,721,432,024 bytes | `Lightricks/LTX-2.5` @ `8a4ff96f581e72bedc1b44367581c49d544a05f1` | `f9c4c2ae9a6aa8f732eb02a1c4c3b34888caad3dd35bb65deaf3b5043cda78fa` | Distilled NVFP4 DiT, 7876 tensors | The same path at `6c7e5e57...` is a different artefact, and the next section gives both value sets |
| LTX-2.5 distilled LoRA | `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 8,899,889,568 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` | n/a (non-quantized) | REQUIRED by every non-distilled two-stage recipe — `ti2vid_two_stage`, `keyframe_interpolation`, `a2vid_two_stage`, `res2s_two_stage` and `dfr` — and applied to both stages on the last two; rank and alpha 450; version 2.5.0 | A load that omits it on those five arms is refused by name; distinct from the 327,322,640-byte IC-LoRA |
| LTX-2.5 video VAE | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | 1,452,269,922 bytes | `Lightricks/LTX-2.5` @ `8a4ff96f581e72bedc1b44367581c49d544a05f1` | `685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d` (non-quantized; hashed anyway, see the note above this table) | The `--video-vae` argument of every render; the CONV VAE, which is what the shipped recipes pass | The DiffVAE sibling `ltx-2.5-video-vae-bf16.safetensors` is refused by name rather than silently downgraded |
| LTX-2.5 audio VAE | `vae/ltx-2.5-audio-vae-bf16.safetensors` | 364,866,540 bytes | `Lightricks/LTX-2.5` @ `8a4ff96f581e72bedc1b44367581c49d544a05f1` | `c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5` (non-quantized; hashed anyway, see the note above this table) | The `--audio-vae` argument of every render | No quantized arm is recorded |
| LTX-2.5 Gemma-4 12B text encoder, bf16 | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | 26,263,858,182 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` (gated) | `ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1` (non-quantized; hashed anyway, and derived three independent times — the download's `x-linked-etag`, a CIFS read, and the worker's local disk during the render) | The **upstream oracle's** text tower, and the only one it accepts: `tools/oracle/ltx2_oracle.py` and #1864's reference render. This project's loader now reads it too: its two caption projections are stored BF16 [4096, 188160] and [2048, 188160] with no scale tensor in the file, and until [#2140](https://github.com/mudler/vllm.cpp/issues/2140) the loader doubled that already logical width to 376320 and refused. That refusal was MEASURED and LOCALISED on 2026-08-27 (`rc` job `001c36e9`): the 12 B tower itself loaded in bf16 in 34.815 s, and only the two caption projections refused. Measured on these bytes, not inferred | Unlike the torchao row below, this file DOES carry a `__metadata__` block, so `--encoder-config` is not required beside it. Our renders still take the NVFP4 torchao tower in the row below: no render has yet been gated on this one, and #1854's arm-matched comparison is what will do it. Upstream reads no torchao tensor at pin `fd4ded7f`, so the two are not interchangeable in either direction |
| LTX-2.5 Gemma-4 12B text encoder | `text_encoders/gemma4-12b-with-proj-nvfp4-torchao.safetensors` | 7,423,624,178 bytes | `vonkaiser/LTX-2.5-FP8-NVFP4` @ `5a40ba9ab209a90ddb7943d1e3d374c51cfd3256` | `12132b7157925332d2b21de9fc6f507c14f4f0cbc7081484d1968ebf8a19b4bf` | The `--encoder` argument of every render, NVFP4 torchao | This file carries NO `__metadata__` block, so `--encoder-config` is REQUIRED beside it and the loader refuses by name without it (`ltx2_text_encoder.cpp`) |
| Qwen3.8-27B GGUF language model | `Qwen3.8-27B-Q4_K_M.gguf` | 17,106,775,008 bytes | `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | Q4_K_M text model loads through `--model` and decodes on CPU | **The token gate against llama.cpp `b10451` FAILED** on 2026-08-23: tokenizer exact 6/6, generation divergent 5/6 ([evidence](bench-evidence/qwen38-27b-q4km-token-gate-20260823.md), #821). GGUF multimodal forward is missing |
| Qwen3.8-27B GGUF projector | `mmproj-BF16.gguf` | 931,146,432 bytes | `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` | BF16 `clip` projector loads and validates through `--mmproj` | No request path runs the loaded projector |
| Qwen3.8-27B mixed FP8 and NVFP4 | `model.safetensors` | 22,568,192,096 bytes | `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | `c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05` | NVFP4 modules load | FP8 modules and quantized KV cache are refused |
| Qwen3.8-27B MTP drafter | `model_mtp.safetensors` | 849,400,392 bytes | `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | n/a (non-quantized) | BF16 MTP artifact is present | MTP execution is owed |
| Qwen3.8-27B ModelOpt NVFP4 shard 1 of 4 | `model-00001-of-00004.safetensors` | 9,965,644,108 bytes | `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a22990e82c54c1d48ee77c491b87825680` | Locally computed hash is owed; the bytes are not mirrored here, and the four shards are verified semantically instead (header parse plus data end equal to the published size); #821 | 208 per-tensor static FP8 and 193 W4A16_NVFP4 modules load | The declared FP8 KV cache is unread; #1593 |
| Qwen3.8-27B ModelOpt NVFP4 shard 2 of 4 | `model-00002-of-00004.safetensors` | 9,985,743,924 bytes | `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a22990e82c54c1d48ee77c491b87825680` | Locally computed hash is owed; #821 | Same arms as shard 1 | The declared FP8 KV cache is unread; #1593 |
| Qwen3.8-27B ModelOpt NVFP4 shard 3 of 4 | `model-00003-of-00004.safetensors` | 1,120,886,516 bytes | `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a22990e82c54c1d48ee77c491b87825680` | Locally computed hash is owed; #821 | Same arms as shard 1 | The declared FP8 KV cache is unread; #1593 |
| Qwen3.8-27B ModelOpt MTP drafter | `model-00004-of-00004.safetensors` | 849,400,592 bytes | `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a22990e82c54c1d48ee77c491b87825680` | Locally computed hash is owed; #821 | Fifteen BF16 MTP tensors are present and unquantized | MTP execution is owed |
| Qwen3.8-2.4T-A95B | `UD-Q1_0` ten-file GGUF split | about 370 GiB | `unsloth/Qwen3.8-2.4T-A95B-GGUF` @ `567d3e6ac26c5474b18311e619c04350fb9a5556` | `b7770552b2ac24e7334c917bc92e90e218e87cfe29484db65e62e8ef2a60334d` (shard 1); `2765517f833c736338d3ab34354e1c10eb8d79e62325f998285b435e5cf03dcd` (shard 2) | CPU expert streaming from disk | CUDA refuses a checkpoint that exceeds device capacity |
| Llama-3.2-1B-Instruct EXL3 3.0bpw (the first EXL3 checkpoint that GENERATES) | `model.safetensors` | 1,089,087,416 bytes | `turboderp/Llama-3.2-1B-Instruct-exl3` @ `f8f438c290680b15622270eff03bef23a458b1cf` (revision `3.0bpw` -- this repo publishes ONE BRANCH PER BIT WIDTH and `main` carries no weights at all, so a bare repo id resolves to nothing) | `3c0341e9c7c4c16a86a499de1dff4f6d7de9855541d669f3b0e214d72b54c2fc` | LOADS and GENERATES end to end through `vllm-cli` on `--device cpu`: `The capital of France is` -> ` Paris. Paris is known for its famous landmarks such as the Eiffel Tower` (greedy, 16 tokens, 2026-08-28). Native exllamav3 layout, no `.rank{r}` slicing; the body is 3-bit and `lm_head` is SIX-bit, resolved per tensor | Codebook **0** (the original QTIP 3INST), because the artifact ships no `mcg` marker and `LinearEXL3` derives the codebook from tensor PRESENCE. The CUDA arm instantiates codebook 1 only and REFUSES this checkpoint by name, so it runs on a CPU queue today (0.040 tok/s at 16 tokens; no speed claim is made on any axis and none is intended). q/k/v and gate/up run as separate GEMMs rather than one merged operand |
| DeepSeek-V4-Flash EXL3 trellis shard 1 of 172 | `exl3-layer-000-tp4-rank0.safetensors` | 515,850,920 bytes | `0xSero/deepseek-v4-flash-0731-spark` @ `22f28d32b9b29b4352eaa380ff8c2c170b2847ab` | `2ed7ae798a794019810b027fe2609e2cf4ad78d70b49c47b2970d03a0a7aaadf` | The rank-sliced EXL3 routed-expert tower LOADS (TP4 coalesced to TP1) and its experts EXECUTE through `vt::Exl3Gemm` on a CPU queue | The CUDA arm compiles for `sm_121a` and its numeric gates PASSED on GB10 on 2026-08-28 (`had_r_128` byte-identical, `exl3_gemm` `rel_rms 5.538e-4`, GEMV tier 3c `5.160e-4`); the FUSED MoE device arm still cannot run, because it needs a device-resident tower, so the routed experts execute on a CPU queue. That run decoded ZERO tensors of THIS artifact -- it found no readable shard -- so nothing here is a claim about these weights on a device. A SYNTHETIC rank-sliced checkpoint now loads and emits logits end to end; THIS artifact still does not, because its DSA compressor and indexer tensors are stored at twice the width the host forward indexes (`compressor.wgate` `[2*head_dim, H]`) and the loader refuses them BY NAME, and because its tokenizer is not read ([#1924](https://github.com/mudler/vllm.cpp/issues/1924)) |
| DeepSeek-V4-Flash EXL3 carried tower shard 1 of 5 | `carried-001.safetensors` | 4,288,630,252 bytes | `0xSero/deepseek-v4-flash-0731-spark` @ `22f28d32b9b29b4352eaa380ff8c2c170b2847ab` | `3b67ae29f1e75c2ecadfcafd3b0eecec640b06fd60b832f77e6bd3c2a8c85ccf` | The un-requantized `deepseek_v4_fp8` attention, router, shared-expert, compressor and embedding tensors, MATERIALIZED at load into the host-float tower the forward composes with — block-wise FP8 (`F8_E4M3` + `F8_E8M0` over 128x128 blocks) decoded to f32, BF16 norms and embeddings widened, I64 `tid2eid` narrowed to int32 | The DSA compressor and indexer tensors of this artifact are `2 * head_dim` / `2 * index_head_dim` wide and the loader refuses them by name (41 of its 43 layers carry a compressor); the 3,985 `mtp.*` NVFP4 draft tensors are skipped and counted, never silently dropped |
| GLM-5.3-Flash FP8 source | `model-000{01..62}-of-00062.safetensors` | 328,326,771,576 bytes total (305.78 GiB) | `zai-org/GLM-5.3-Flash` @ `main`, read 2026-08-26 | Owed: no byte of payload has been fetched, so no local hash exists to state, and an unauthenticated tree hash is not a pin here | Declared source of `scripts/convert-glm5-next-gguf.py`. Only the safetensors HEADERS were read, by HTTP RANGE over all 62 shards: 76,108 tensors, `F8_E4M3` block-quantized at `weight_block_size: [128, 128]` with `weight_scale_inv` companions, plus BF16 and F32 scales | **Nothing has been converted.** The download needs explicit developer authority and a box with room for 305.78 GiB of source and ~100.35 GiB of output at once; owed as O7 on [#2011](https://github.com/mudler/vllm.cpp/issues/2011). The revision is a branch name and not a commit, which is NOT a pin: it is what was read, and W7b re-reads and records the commit when it stages the bytes |
| GLM-5.3-Flash GGUF | none exists | n/a | `unsloth/GLM-5.3-Flash-GGUF`, `AtomicChat/GLM-5.3-Flash-GGUF`, `aj9o9/GLM-5.3-Flash-GGUF`, `vcruz305/GLM-5.3-Flash-GGUF`, all read 2026-08-26 | n/a | none | **All four repositories named `*-GGUF` contain ZERO `.gguf` files** — READMEs, a `.gitattributes` and four PNGs between them. A repository name is not an artifact, and this row exists so the next reader does not go looking again. llama.cpp cannot produce one either: no `glm5_next` at `origin/master` `539f24529` or at our pin `b10451` |
| GLM-5.3-Flash config | `config.json` | 69,416 bytes | `zai-org/GLM-5.3-Flash` @ `main`, read 2026-08-27 | sha256 `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f` | The ONLY byte of this checkpoint any change on this row has consumed. Checked in verbatim as `tests/vllm/models/fixtures/glm5_next/config.json` and used as W1's gate fixture, so the config layer is gated against what the checkpoint says rather than against what a port's author believed it says | **Arms refused by name:** every arm. `Glm5NextForConditionalGeneration` is REGISTERED and its config RESOLVES; the weight loader, the forward and the KV-cache spec all refuse, naming the wave that owes each ([#2067](https://github.com/mudler/vllm.cpp/issues/2067)). The revision is a branch name and not a commit, which is NOT a pin for the WEIGHTS; for this one file the sha256 above is the pin |
| Qwen3.5-0.8B (Tenstorrent P150 arm) | `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 bytes | `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`, authorized 2026-08-23 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` (non-quantized; hashed anyway from the local bytes the gates and the eager profile consumed) | bf16 on the Tenstorrent P150: the sacred greedy pair, both ambient legs, and the #1715/#2107 profile legs all ran from this snapshot | **Arms refused by name:** GGUF k-quant arms on TT — no TT kernels exist for them, refused at load; Qwen3.8-27B on TT — no arm fits the P150 (bf16 53.8 GB), refused at load |
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

**The file it writes is now OPENED by this tree, and it still does not load.**
W1 ([#2067](https://github.com/mudler/vllm.cpp/issues/2067)) gave `glm5next` its
`general.architecture` dispatch row and registered
`Glm5NextForConditionalGeneration`, so passing such a file to a `.gguf` entry
point reads its metadata, cross-checks its per-layer schedule against its tensor
inventory, and validates its config — through the same parser a `config.json`
descends through. It then **refuses by name** at weight materialization, because
no weight tower, forward or KV-cache spec is ported (W5 owes them). What changed
is the refusal: it names the wave that owes the work instead of reporting the
file's architecture as unrecognized.

**And no artifact exists to try it on.** The converter has never been run
against the real 305.78 GiB checkpoint; that needs explicit developer authority
for the download and a box with room for source and output at once
([#2011](https://github.com/mudler/vllm.cpp/issues/2011)).

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
