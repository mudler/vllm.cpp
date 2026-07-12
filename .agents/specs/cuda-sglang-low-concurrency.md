# Spike: CUDA low-concurrency serving versus SGLang

**Owning rows:** `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` for the independently
claimable harness/checkpoint work; `BACKEND-GATE-CUDA-SGLANG` for the binding
cache-neutral campaign after async serving closes; and
`BACKEND-GATE-CUDA-SGLANG-PREFIX` for the distinct deterministic shared-prefix
cache-on gate.
**Recommendation:** the implemented preflight row remains `GATING`; the
cache-neutral binding gate is `BLOCKED` until `SERVE-ASYNC-LLM` provides real
incremental HTTP streaming. The fully scoped shared-prefix row is `READY` for
harness/instrumentation work, while its binding run is dependency-blocked on
exact SGLang v0.5.15 equivalence, `KV-MAMBA-ALIGN`, and async serving.

This is the leaf spike required by the
[competitive benchmark matrix](competitive-benchmarks.md) and the
[benchmark protocol](../benchmark-protocol.md). It fixes the competitor,
checkpoint-equivalence rule, workload, artifacts, and independent work
packages before anyone installs SGLang or consumes the shared GPU.

## Scope

| In scope | Out of scope |
|---|---|
| NVIDIA CUDA on `dgx.casa` (GB10, sm121, aarch64) | Generic large-concurrency throughput gates, which remain owned by vLLM parity; the dedicated shared-prefix extension intentionally includes c32 |
| Qwen3.6 27B dense and 35B-A3B MoE text-only serving | Vision/audio inputs and multimodal encoder performance |
| Concurrency `1,2,4,8,16`, production graphs, streaming completions | SGLang tuning that changes model semantics or disables production features |
| Exact shared checkpoint/quantization first; vLLM and vllm.cpp arms always present | Ratios between different checkpoints, architectures, or converted weights |
| Correctness, token counts, every latency/throughput axis, peak memory, nsys | Declaring SGLang a floor before equivalence is demonstrated |

The endpoint is `POST /v1/completions`, not chat, so chat-template differences
cannot contaminate the comparison. All arms are language-only, greedy,
`ignore_eos=true`, and use the same prompt text generated from and verified
against one tokenizer snapshot.

## Upstream chain and pins

| Component | Immutable pin | Relevant source/decision |
|---|---|---|
| SGLang source | tag `v0.5.13`, commit `28b095c01005d4a3a2a5b637b7d028b07fba31b2` | [release](https://github.com/sgl-project/sglang/releases/tag/v0.5.13), [tag commit](https://github.com/sgl-project/sglang/commit/28b095c01005d4a3a2a5b637b7d028b07fba31b2) |
| SGLang runtime image | `docker.io/lmsysorg/sglang:v0.5.13-cu130-runtime@sha256:9631280f57d95503ed64cf3892de72190aafbfe6e58e90718a019fa775113bfb` | Official release workflow emits the CUDA 13 runtime tags ([workflow lines 42-54](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/.github/workflows/release-docker-runtime.yml#L42-L54)); the reusable workflow builds amd64 and arm64 then joins them ([lines 1-4](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/.github/workflows/_docker-build-and-publish.yml#L1-L4), [250-328](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/.github/workflows/_docker-build-and-publish.yml#L250-L328)). Docker Hub API inspection on 2026-07-10 reported that manifest-list digest with arm64 and amd64 images. |
| vLLM oracle | local pin `e24d1b24fe96a56ba8b0d653efa076d03eb95d6c`, `~/venvs/vllm-oracle` | Production CUDA graphs remain enabled; never use `--enforce-eager` |
| vllm.cpp | commit under test, recorded per run | [server CLI](../../examples/server/main.cpp), [OpenAI transport](../../src/vllm/entrypoints/openai/api_server.cpp) |
| Load client | SGLang `bench_serving` from the same pinned image | It maps both `sglang-oai` and `vllm` to the same OpenAI completions request function ([lines 874-887](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/bench_serving.py#L874-L887)) |
| Shared-prefix candidate | SGLang tag `v0.5.15`, commit `f63458b5beaceabbd9d749b9fc956370e1b649e6`; image `lmsysorg/sglang:v0.5.15-cu130@sha256:d0a667eca4e6fff64f7758c5fb1720e16faa806f90ea767e018bb8fa1b09dd44` | This newer pin applies only to `BACKEND-GATE-CUDA-SGLANG-PREFIX` until the existing P1 harness is deliberately repinned and re-gated; it is never substituted silently into old v0.5.13 evidence. |
| Referenced DGX recipe | `Weschera/qwen-sglang-dgx-spark` commit `03253ef98c01de59a21c85b9a5cc6a27a871c383`; `spark-bench` commit `dac4e108` | The repository withdraws its original 10--40x claim as a cache-on/cache-off mismatch ([README lines 7-12](https://github.com/Weschera/qwen-sglang-dgx-spark/blob/03253ef98c01de59a21c85b9a5cc6a27a871c383/README.md#L7-L12)); its remaining cache-on figures are inputs to reproduce, not accepted evidence. |

The official image is a verified production/runtime artifact, not a floating
`latest` tag. Provisioning must additionally record the resolved
platform-specific digest and OCI `org.opencontainers.image.revision`; a
revision other than `28b095c01005d4a3a2a5b637b7d028b07fba31b2` is a hard
failure even when the manifest-list digest matches.

SGLang's production guidance recommends its runtime image and the CUDA 13
variant for CUDA 13 systems ([install lines 67-99](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/docs/get_started/install.md#L67-L99)).
Its online benchmark guide identifies `bench_serving` as the realistic
scheduler/HTTP test and requires at least five request waves
([benchmark guide lines 5-20](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/docs/developer_guide/benchmark_and_profiling.md#L5-L20)).

## Model and quantization compatibility

### Shared checkpoint candidates

| Model | Exact snapshot on DGX | Shipped format | SGLang source path | Initial classification |
|---|---|---|---|---|
| 27B dense | `unsloth/Qwen3.6-27B-NVFP4` at `890bdef7a42feba6d83b6e17a03315c694112f2a` | `Qwen3_5ForConditionalGeneration`; compressed-tensors `nvfp4-pack-quantized`, dynamic W4A4, group 16 | Qwen3.5 entry class ([model lines 2035-2048](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/srt/models/qwen3_5.py#L2035-L2048)); compressed-tensors W4A4 creates packed weights/scales and dispatches FP4 GEMM ([scheme lines 32-169](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L32-L169)) | Native path exists. It becomes equivalent only after exact load, tensor/scale audit, and 16-prompt token check. |
| 35B-A3B MoE | `nvidia/Qwen3.6-35B-A3B-NVFP4` at `491c2f1ea524c639598bf8fa787a93fed5a6fbce` | `Qwen3_5MoeForConditionalGeneration`; ModelOpt `MIXED_PRECISION`: selected FP8 linears/activations plus W4A16 NVFP4 MoE/shared-expert/lm-head layers; no FP8 KV-cache declaration | SGLang parses non-Nemotron `MIXED_PRECISION` as `w4afp8` ([model-config lines 1017-1033](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/srt/configs/model_config.py#L1017-L1033)); that config uses FP8 linears and a legacy W4A8 MoE layout ([w4afp8 lines 35-105](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/srt/layers/quantization/w4afp8.py#L35-L105)). SGLang also has a per-layer ModelOpt mixed dispatcher ([lines 605-779](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/srt/layers/quantization/modelopt_quant.py#L605-L779)), but its parser selects that path only for Nemotron-H. | Compatibility is unresolved and must be probed. Source alone does not justify an equivalent claim. |

Both architecture IDs are registered, and SGLang explicitly permits
`--language-only` for both Qwen3.5 wrappers
([server validation lines 3972-4020](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/srt/server_args.py#L3972-L4020)).
That establishes model-family and text-only support; it does not establish that
these two particular NVFP4 snapshots load with the same quantization semantics.

### Exact-loader preflight

For each snapshot, launch with **no `--quantization` override** first. The
checkpoint's own metadata must select the path. Retain:

1. `config.json`, `hf_quant_config.json` when present, safetensors index, and a
   sorted `sha256sum` manifest of every resolved checkpoint file.
2. Complete startup log, `/server_info`, selected architecture, reported
   quantization, weight-loader warnings, unexpected/missing tensors, and the
   kernel names from the trace.
3. For 27B, proof that packed weight/global/group/input scales map to
   `CompressedTensorsW4A4Fp4`, including the reciprocal-global-scale convention
   documented by [the existing W4A4 spike](qwen27b-w4a4-notes.md).
4. For 35B, proof that FP8 and W4A16 NVFP4 layers follow the checkpoint's
   per-layer map. A successful process start routed through generic `w4afp8` is
   not sufficient evidence because its group size and MoE activation format
   differ in source.

An explicit quantization flag may be tried only as a separately labeled
diagnostic after the automatic load fails. It is not binding unless it consumes
the unchanged files and demonstrably selects the same per-layer algorithms and
scale conventions.

### Fallback policy

The fallback order is strict:

1. Exact 27B and exact 35B snapshots above.
2. Another immutable, upstream-published Qwen3.5/3.6 text checkpoint that all
   three engines consume **unchanged**, with the same architecture and native
   quantization semantics. Add its repository revision and file manifest before
   running it.
3. A locally converted or re-quantized checkpoint may be run only under the
   label `converted-nonbinding` to diagnose scheduler/kernel behavior.

There is no ratio across different checkpoints. A SGLang result on a different
model cannot be divided by a vLLM/vllm.cpp gate-model result. Converted results
remain non-binding under this spike even if their short greedy outputs match;
changing that requires a later quality-equivalence gate and policy update.
If only the exact 27B is shared, SGLang can become a binding 27B floor while the
35B SGLang row remains open and informational.

## Our baseline and known gaps

| Surface | Current anchor/evidence | Consequence for this gate |
|---|---|---|
| 27B exact model and compressed-tensors W4A4 | [27B gate](../../tests/parity/test_qwen27_paged_engine.cpp), [quant row](../quantization-matrix.md) | Correctness and production offline throughput exist; online low-concurrency evidence does not. |
| 35B exact model and ModelOpt W4A16/FP8 mix | [35B gate](../../tests/parity/test_qwen36_paged_engine.cpp), [quant row](../quantization-matrix.md) | Same: model correctness exists, low-concurrency serving does not. |
| OpenAI completion endpoint | [server CLI](../../examples/server/main.cpp), [API server](../../src/vllm/entrypoints/openai/api_server.cpp) | The request schema is usable by the common client. |
| Concurrent HTTP requests | [engine mutex](../../src/vllm/entrypoints/openai/api_server.cpp#L18) serializes a whole request | Concurrency above one currently exposes a real serving gap rather than dynamic batching. |
| Streaming timing | [serving contract](../../include/vllm/entrypoints/openai/serving_completion.h#L5) runs the engine to completion and precomputes all SSE chunks; [transport](../../src/vllm/entrypoints/openai/api_server.cpp#L180) then writes them | Client-observed TTFT is full-generation latency and ITL is transport drain time. Those axes are invalid until `SERVE-ASYNC-LLM` lands real incremental generation/SSE. |
| Stream usage | [deferred surface](../../include/vllm/entrypoints/openai/serving_completion.h#L14) | Not a blocker: the harness has an independent non-streaming token-count preflight. It must not silently trust requested length as observed length. |
| Prior online campaign | [`SERVE-GATE-ONLINE`](../engine-matrix.md) records startup failures/aborts | No prior number is a baseline or denominator for this campaign. |

Thus `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` can be `READY` without async
serving. The binding `BACKEND-GATE-CUDA-SGLANG` remains `BLOCKED` on both a
successful exact-equivalence preflight and `SERVE-ASYNC-LLM`.

## Port and harness map

| Upstream/reference | Local deliverable | Purpose/deviation |
|---|---|---|
| SGLang `benchmark/datasets/custom.py` ([lines 54-147](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/benchmark/datasets/custom.py#L54-L147)) | `tools/bench/make_serve_low_corpus.py` | Emit deterministic, disjoint JSONL partitions whose prompt text re-encodes to exactly 1024 IDs; also emit token IDs and hashes for independent checking. |
| vllm.cpp tokenizer CLI | Extend or script [examples/tokenize](../../examples/tokenize/main.cpp) in the harness, not the runtime | Verify every prompt with both the HF tokenizer in the pinned client and our tokenizer. Any ID mismatch voids the corpus. |
| SGLang OpenAI request function ([lines 232-344](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/bench_serving.py#L232-L344)) | `tools/bench/run_serve_low.py` | Invoke the unmodified pinned client for each endpoint; add orchestration, readiness, owned-PID cleanup, and artifact metadata outside the timed interval. |
| SGLang metric calculation/output ([lines 968-1140](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/bench_serving.py#L968-L1140), [1588-1675](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/bench_serving.py#L1588-L1675)) | `tools/bench/summarize_serve_low.py` | Preserve raw JSONL and compute missing P90/P99 axes, repetition spread, ratios, and precondition failures from raw per-request arrays. Do not patch competitor code in-place. |
| Linux `/proc/*/smaps_rollup` and cgroup process list | `tools/bench/sample_process_memory.py` | Sample aggregate server-tree PSS/RSS and system available-memory delta at 100 ms; emit JSONL. This is the common GB10 unified-memory measurement. |
| Benchmark protocol and GPU skill | `scripts/dgx-sglang-low-concurrency.sh` | One reproducible campaign driver: no pull/install, one `flock`, sequential arms, traps that stop only owned PIDs/containers. |
| SGLang Nsight method ([lines 296-345](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/docs/developer_guide/benchmark_and_profiling.md#L296-L345)) and project trace directive | campaign profile subcommand | Start system-wide CUDA-only `nsys` after server warmup while the GPU lock guarantees exclusivity; capture SGLang, vLLM, and ours on identical trace partitions. |

The benchmark client stays byte-for-byte from the pinned image. Local code
wraps it and validates its JSON; it does not change request timing or metric
formulas before seeing results.

## Canonical corpus and requests

The corpus generator creates a warmup set and 15 measured partitions per
model: `5 concurrency points * 3 repetitions`. Each measured partition has 80
unique prompts, so every point has at least five complete waves:

| Concurrency | Requests per repetition | Minimum complete waves | Repetitions |
|---:|---:|---:|---:|
| 1 | 80 | 80 | 3 |
| 2 | 80 | 40 | 3 |
| 4 | 80 | 20 | 3 |
| 8 | 80 | 10 | 3 |
| 16 | 80 | 5 | 3 |

Corpus invariants:

- Every prompt encodes to exactly 1024 token IDs with the snapshot tokenizer,
  the SGLang client tokenizer, the vLLM tokenizer, and vllm.cpp's tokenizer.
- The JSONL stores `prompt_sha256`, canonical token IDs, prompt text, tokenizer
  revision, and expected length. The SGLang `custom` shape contains two
  conversation turns; the second is ignored because output is fixed at 128.
- Warmup prompts and all `(concurrency, repetition)` partitions are disjoint.
  No measured prompt is repeated on a server, and pairwise common token
  prefixes are bounded to 32 tokens so radix/prefix cache hits cannot dominate.
- The same partition SHA and order are used by all three arms. Seed is `0`.
- Output is exactly 128 generated tokens, greedy (`temperature=0`, `top_p=1`),
  `ignore_eos=true`, streaming, one-token stream interval, no speculative
  decoding, no LoRA, and no prefix-cache flush inside a timed run.
- Server context length is 1152. The production cache has at least 20,480 token
  slots, enough for all 16 requests; startup logs must record the actual usable
  capacity. Different native page layouts are allowed, but capacity starvation
  is not.

Use one fixed concurrency order in all arms for a repetition: ascending for
rep 1, descending for rep 2, and `4,1,16,2,8` for rep 3. This makes warming and
thermal order reproducible without reusing prompts.

### Correctness and token-count precondition

Before timed runs for each model/engine:

1. Run an **untimed engine-native token-ID preflight** on the same stored 1024
   `input_ids`. For SGLang, call `POST /generate` with those IDs and greedy
   `sampling_params` (`max_new_tokens=128`, `temperature=0`, `top_p=1`,
   `ignore_eos=true`) and retain its native `output_ids`; the pinned server
   returns them directly ([HTTP handler lines 714-718](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/srt/entrypoints/http_server.py#L714-L718), [response lines 2640-2660](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/srt/managers/tokenizer_manager.py#L2640-L2660)). Capture native output IDs from the pinned vLLM engine and the vllm.cpp direct engine/gate path too.
2. Require exactly 128 output IDs. Ours must be 16/16 token-for-token with the
   pinned vLLM oracle. SGLang must also be 16/16 for its result to become
   binding; divergence leaves it an informational leader and triggers
   logits/quant-path diagnosis. Detokenizing and re-tokenizing returned text is
   only a text-equivalence diagnostic and never counts as token-ID parity.
3. Separately submit the same fixed subset through each OpenAI-compatible HTTP
   path non-streaming. Require HTTP 200, `finish_reason=length`,
   `usage.prompt_tokens=1024`, `usage.completion_tokens=128`, and total `1152`
   for every request. This validates serving shape, not token identity.
4. Send a short streaming probe and require the first data chunk before the
   request completes plus 128 token emissions at `stream_interval=1`. This is
   expected to fail for today's buffered vllm.cpp server and is the explicit
   dependency on `SERVE-ASYNC-LLM`.

Any failed request, length mismatch, buffered stream, tokenizer mismatch, or
different checkpoint manifest voids every timed number from that arm.

## Exact client command

The campaign mounts the evidence and model-repository roots read-only into a
CPU-only client container. For each engine, concurrency, and repetition it runs
the same pinned module; only `BACKEND` and `BASE_URL` change:

```bash
docker run --rm --network=host --pull=never \
  --mount type=bind,src="$MODEL_REPO",dst=/models/gate,readonly \
  --mount type=bind,src="$CLAIM/evidence",dst=/evidence \
  "$SG_IMAGE" \
  python3 -m sglang.bench_serving \
    --backend "$BACKEND" \
    --base-url "$BASE_URL" \
    --model gate \
    --tokenizer "/models/gate/snapshots/$MODEL_REV" \
    --dataset-name custom \
    --dataset-path "/evidence/corpus/$MODEL_KEY/c${C}-r${REP}.jsonl" \
    --num-prompts 80 \
    --sharegpt-output-len 128 \
    --request-rate inf \
    --max-concurrency "$C" \
    --warmup-requests 0 \
    --seed 0 \
    --temperature 0 \
    --top-p 1 \
    --extra-request-body '{"ignore_eos":true}' \
    --output-details \
    --output-file "/evidence/raw/$MODEL_KEY/$ENGINE/c${C}-r${REP}.jsonl"
```

Use `BACKEND=sglang-oai` for SGLang and `BACKEND=vllm` for both OpenAI-compatible
vLLM and vllm.cpp endpoints. Both names call the same client function; this
avoids a client implementation confound. Warmup is performed by the campaign
with its disjoint corpus before this command, so the client's built-in warmup
is zero.

## Server arms

The commands below are templates filled with exact local snapshot paths and
recorded verbatim by the campaign. They use production graph defaults. Do not
add an eager/graph-disabling flag to a measured arm.

### SGLang

```bash
docker run --rm --gpus all --ipc=host --network=host --pull=never \
  --name "sglang-low-${MODEL_KEY}" \
  --mount type=bind,src="$MODEL_REPO",dst=/models/gate,readonly \
  --mount type=bind,src="$CLAIM/sglang-cache/$MODEL_KEY",dst=/root/.cache/sglang \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  "$SG_IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "/models/gate/snapshots/$MODEL_REV" \
    --served-model-name gate \
    --host 127.0.0.1 --port 30000 \
    --language-only \
    --context-length 1152 \
    --max-running-requests 16 \
    --max-total-tokens 20480 \
    --mem-fraction-static 0.85 \
    --stream-interval 1
```

There is intentionally no quantization flag. Any model-specific flag added
after a failed preflight creates a separately labeled diagnostic arm.

### vLLM oracle

```bash
~/venvs/vllm-oracle/bin/vllm serve "$MODEL_SNAPSHOT" \
  --served-model-name gate \
  --host 127.0.0.1 --port 30000 \
  --language-model-only \
  --max-model-len 1152 \
  --max-num-seqs 16 \
  --block-size 32 \
  --num-gpu-blocks-override 640 \
  --gpu-memory-utilization 0.85 \
  --stream-interval 1
```

This is the fresh compatibility denominator on every run. CUDA graphs stay on.

### vllm.cpp

```bash
"$VLLM_CPP_BUILD/examples/server" \
  --model "$MODEL_SNAPSHOT" \
  --served-model-name gate \
  --host 127.0.0.1 --port 30000 \
  --block-size 32 \
  --num-blocks 640 \
  --max-model-len 1152
```

The local server currently has no max-running-request or stream-interval CLI
because it serializes complete requests. The binding campaign waits for the
async serving implementation and must record its new equivalent controls.

## Provisioning versus the one GPU lock

Provisioning is a separate, non-GPU maintenance step. It must be scheduled when
no benchmark series is active because image pulls and decompression create disk,
network, CPU, and page-cache contention. It never runs inside the timed campaign
and never mutates `~/venvs/vllm-oracle`:

```bash
export SG_IMAGE='docker.io/lmsysorg/sglang:v0.5.13-cu130-runtime@sha256:9631280f57d95503ed64cf3892de72190aafbfe6e58e90718a019fa775113bfb'
docker pull "$SG_IMAGE"
docker image inspect "$SG_IMAGE" > "$CLAIM/provision/image-inspect.json"
docker run --rm --pull=never "$SG_IMAGE" \
  python3 -c 'import importlib.metadata as m; print(m.version("sglang"))' \
  > "$CLAIM/provision/sglang-version.txt"
```

The execution script accepts only the digest-pinned image and uses
`--pull=never`. JIT/graph warmup is execution, so it occurs under the lock and
is excluded from timed/profile ranges. The complete two-model, three-engine,
three-repetition, memory, and nsys series owns **one** mutex acquisition:

```bash
setsid flock /tmp/gpu -c \
  "$CLAIM/run-campaign.sh --image '$SG_IMAGE' --pull never" \
  > "$CLAIM/evidence/campaign.log" 2>&1
```

The script uses PID/container-specific traps and stops only processes it
started. No `pkill`, Docker-wide cleanup, polling loop, second `flock`, or
interleaved GPU claim is permitted.

## Metrics and artifacts

| Class | Required fields |
|---|---|
| Higher is better | request/s; input, output, and total token/s |
| Lower is better | mean/P50/P90/P99 E2E, TTFT, and TPOT; mean/P50/P90/P95/P99/max ITL |
| Capacity/health | completed requests, errors, achieved/peak concurrency, input/output counts, corpus/checkpoint hashes |
| Memory | ready-state and peak aggregate PSS/RSS for the server cgroup/process tree; peak system `MemAvailable` drop; numeric accelerator resident memory when available |
| Reproduction | all three raw repetitions, mean/median/range, coefficient of variation, ours/vLLM, ours/SGLang, and ours/best-floor ratios |

The GB10 reports unified memory and returned `N/A` for `nvidia-smi` total/used
memory during the 2026-07-10 source spike. Therefore the binding common metric
is a 100 ms aggregate of `/proc/<pid>/smaps_rollup` PSS across the server's
process tree/cgroup plus host `MemAvailable` delta. `nvidia-smi` samples are
still retained when numeric; they are not silently converted to zero. The
memory sampler starts after the server is ready, records an idle baseline, and
runs through warmup and the measured point. Startup peak is reported separately.

SGLang's JSON output includes request totals and per-request TTFT/ITL/generated
text when `--output-details` is set
([output lines 1588-1675](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/bench_serving.py#L1588-L1675)).
The local summarizer recomputes every percentile from those raw arrays and
stores its formulas/version. Requested output length is never accepted as the
only token-count proof; the non-streaming usage preflight is mandatory.

Evidence root:

```text
~/work/vllm.cpp-sglang-low-c/evidence/<vllm.cpp-sha>/
  manifest.json
  provision/{image-inspect.json,sglang-version.txt}
  corpus/<model>/{manifest.json,warmup.jsonl,c*-r*.jsonl,trace-*.jsonl}
  preflight/<model>/<engine>/{requests.jsonl,responses.jsonl,tokens.json}
  raw/<model>/<engine>/c*-r*.jsonl
  memory/<model>/<engine>/c*-r*.jsonl
  logs/<model>/<engine>.log
  nsys/<model>/<engine>/c{1,16}.{nsys-rep,cuda-kernels.csv}
  summary/{all-runs.json,ratios.json,report.md}
```

`manifest.json` includes host idle proof, CPU/GPU/tool versions, environment,
image/source/engine commits, build flags, full commands, model/tokenizer file
hashes, cache capacity, warmup recipe, start/end UTC timestamps, and every child
artifact SHA256.

## Nsight execution trace

After each server has completed its normal warmup, capture CUDA kernel traces on
separate, disjoint 16-request partitions at concurrency 1 and 16, input/output
1024/128. The paired SGLang/ours traces are mandatory for the competitor
comparison; vLLM is also captured because the project protocol requires a
fresh vLLM/ours execution-chain comparison.

Use host system-wide CUDA tracing so the identical command works for a Docker
server and host servers. The one GPU lock guarantees that no unrelated CUDA
process appears. Start `nsys` only after warmup, run the trace client, send
`SIGINT` only to the owned nsys PID, wait for report finalization, then emit the
kernel summary:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --duration=900 -o "$TRACE_BASE" &
NSYS_PID=$!
sleep 3
run_trace_client "$ENGINE" "$MODEL_KEY" "$C"
kill -INT "$NSYS_PID"
wait "$NSYS_PID"
nsys stats --report cuda_gpu_kern_sum --format csv \
  "$TRACE_BASE.nsys-rep" > "$TRACE_BASE.cuda-kernels.csv"
```

The wrapper must fail if the client ends after the 900-second guard, if nsys
does not finalize, or if the trace contains no CUDA kernels. Kernel names and
launch counts drive follow-up source inspection; percentages from a
capture/JIT-contaminated trace do not.

## Tests to port and add

| Source test/spec | Local test | Required behavior |
|---|---|---|
| SGLang `test_bench_serving_functionality.py` ([file](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/test/registered/bench_fn/test_bench_serving_functionality.py)) | `tests/tools/test_serve_low_client.py` | Mock OpenAI SSE server: request body, concurrency cap, error propagation, raw detail retention. |
| SGLang native generation response ([lines 2640-2660](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/srt/managers/tokenizer_manager.py#L2640-L2660)) | `tests/tools/test_serve_low_token_ids.py` | Native `output_ids` are captured without text round-trip, exactly 128 IDs are required, and any 16-prompt mismatch prevents a binding result. |
| SGLang custom dataset loader ([lines 54-147](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/benchmark/datasets/custom.py#L54-L147)) and vLLM custom-seed tests ([file](https://github.com/vllm-project/vllm/blob/e24d1b24fe96a56ba8b0d653efa076d03eb95d6c/tests/benchmarks/test_custom_dataset_seed.py)) | `tests/tools/test_serve_low_corpus.py` | Same seed/partition is byte-identical; different partitions are disjoint; every prompt has 1024 IDs; common-prefix bound and manifest hashes hold. |
| SGLang metric calculation ([lines 968-1140](https://github.com/sgl-project/sglang/blob/28b095c01005d4a3a2a5b637b7d028b07fba31b2/python/sglang/bench_serving.py#L968-L1140)) | `tests/tools/test_serve_low_summary.py` | Hand-computed req/s, tok/s, TTFT, TPOT, ITL percentiles, repetition spreads, axis-wise best-floor ratios, and void-result propagation. |
| vLLM streaming completion tests ([lines 271-284](https://github.com/vllm-project/vllm/blob/e24d1b24fe96a56ba8b0d653efa076d03eb95d6c/tests/entrypoints/openai/completion/test_completion.py#L271-L284)) | Extend [API-server tests](../../tests/vllm/entrypoints/openai/test_api_server.cpp) under `SERVE-ASYNC-LLM` | Streaming begins before generation completion, chunks concatenate to non-stream output, one-token interval, concurrent requests interleave safely. Initially SKIP the timing assertion with reason `SERVE-ASYNC-LLM not implemented`; do not delete it. |
| Linux sampler contract | `tests/tools/test_process_memory_sampler.py` | Synthetic parent/children and short-lived PID cases; valid JSONL; no `N/A -> 0`; peak is monotonic. |
| Full campaign | GPU smoke on `Qwen/Qwen3-0.6B`, then exact gate-model preflights | Digest/revision check, no pull under lock, owned cleanup, all expected raw artifacts, and failed preconditions prevent summary promotion. |

The harness tests are CPU-only and can land before GPU provisioning. The exact
model preflights and full campaign are GPU e2e tiers and always hold the mutex.

## Gates

| Gate | Pass condition |
|---|---|
| Image/source | Manifest digest, platform digest, package version, and OCI revision match the pins above; no floating dependency |
| Model load | Unchanged checkpoint loads language-only with no missing/unexpected quantized tensors and the source-expected quant path |
| Corpus | All four tokenizer paths produce the stored 1024 IDs for every prompt; partition hashes match across arms |
| Correctness | Native engine `output_ids`: ours 16/16 token-exact vs vLLM and SGLang 16/16 before it is binding; OpenAI HTTP separately reports 1024 prompt and 128 completion tokens for every request |
| Serving validity | Streaming first byte precedes request completion; one-token interval; 80/80 successes; no buffered fake TTFT/ITL |
| Reproduction | Three runs per model/engine/concurrency, same binary/config, with spread reported and unexplained outliers rerun |
| Performance | For each model and each concurrency, ours is at least the better equivalent vLLM/SGLang result for req/s and token/s and no worse for every latency percentile |
| Memory | Ours is no worse than every applicable binding floor on the common PSS/available-memory metrics at equal cache capacity |
| Trace | Warmup-excluded CUDA traces and `cuda_gpu_kern_sum` for SGLang, vLLM, and ours at c1/c16, both exact-equivalent models |
| Record | Raw artifacts retained; concise results, commands, ratios, open axes, and evidence path added to the ledger/matrices/README in the same state-changing commit |

The performance floor is axis-wise: if vLLM wins TTFT and SGLang wins TPOT,
ours must meet both. SGLang is binding only for the specific model whose exact
checkpoint and correctness gates passed. "Near parity" is not a pass.

## Dependencies

| Dependency | State/need |
|---|---|
| `SERVE-ASYNC-LLM` | Hard dependency for binding TTFT/ITL and true concurrent online serving; current endpoint buffers the whole result |
| `SERVE-OAI-BASIC` | Existing endpoint is sufficient for harness/preflight; its leaf spike and async transport closure remain separate work |
| `SERVE-STREAM-USAGE` | Not blocking because the harness performs non-stream usage checks; still required for full API parity |
| `BACKEND-GATE-CUDA-VLLM` | Fresh vLLM arm in every campaign; no stale denominator |
| `QUANT-NVFP4-CT-W4A4` | Exact 27B load/compute semantics |
| 35B ModelOpt mixed quant rows | Exact 35B SGLang compatibility must be demonstrated; source ambiguity is not waived |
| Hardware | Idle `dgx.casa`, one `flock /tmp/gpu`, CUDA 13, host `nsys`, at least 200 GB free disk headroom |
| Data | Two immutable HF snapshots above, tokenizer/config/index/shards, deterministic corpus partitions |
| License | SGLang and its benchmark code are Apache-2.0; the harness invokes the official client and ports only test behavior with source attribution |

## Non-overlapping work breakdown

| Order/parallel lane | Owning row/claim | Owned files or evidence | Exit |
|---|---|---|---|
| P1, CPU | `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` harness claim | `tools/bench/*serve_low*`, `scripts/dgx-sglang-low-concurrency.sh`, `tests/tools/test_*serve_low*` | Corpus/client/summary/memory unit tests pass; campaign dry-run creates a complete manifest without GPU use. |
| P1, engine in parallel | `SERVE-ASYNC-LLM` | Engine/serving/HTTP implementation and its ported upstream tests only; no benchmark scripts | Real incremental SSE, in-flight batching, and streaming timing test pass. |
| P1, read-only | compatibility audit under `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` | Checkpoint config/hash report only | Exact per-layer 27B/35B quant expectation frozen before model load. |
| P2, GPU | `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` compatibility claim | `preflight/`, server logs, native output-ID evidence, small nsys evidence | Exact 27B and 35B independently classified equivalent, non-equivalent, or load-blocked. No performance claim yet. |
| P3, GPU | `BACKEND-GATE-CUDA-SGLANG` gate claim after async serving | Full `raw/`, `memory/`, `nsys/`, `summary/` tree | Three-arm c1-16 campaign reproduces and every applicable axis has a decision. |
| P4, integration | root/lead | backend/engine matrices, roadmap, README, porting inventory, ledger, state | Status matches evidence; no converted or mismatched result promoted. |

P1/P2 preflight work and `SERVE-ASYNC-LLM` can run concurrently in separate
worktrees. P3 claims the distinct binding-gate row only after both dependencies
close; an image load alone cannot release it.

## Shared-prefix extension (2026-07-12)

This section is the accepted implementation spike for
`BACKEND-GATE-CUDA-SGLANG-PREFIX`. It does not reinterpret the existing
cache-neutral P1 artifacts and it does not accept the external repository's
numbers. It adds a separately reproducible feature gate because cache-neutral
serving and cache-hit serving exercise different engine paths.

### Source/config audit of the reported win

| Finding | Evidence | Disposition |
|---|---|---|
| The original 10--40x claim was a configuration mismatch. | The recipe itself says every concurrent stream received an identical prompt while SGLang radix caching was on and vLLM prefix caching was explicitly off ([README lines 7-12](https://github.com/Weschera/qwen-sglang-dgx-spark/blob/03253ef98c01de59a21c85b9a5cc6a27a871c383/README.md#L7-L12)); its checked-in vLLM launch still carries `--no-enable-prefix-caching` ([serve-vllm lines 19-24](https://github.com/Weschera/qwen-sglang-dgx-spark/blob/03253ef98c01de59a21c85b9a5cc6a27a871c383/serve-vllm.sh#L19-L24)). | Definitely not an engine-throughput comparison. Reproduce cache-off and cache-on as different gates. |
| The corrected cache-off data slightly favors vLLM. | At 64k/c2,c8,c32 the CSV reports SGLang 25.7/28.6/29.8 output tok/s versus vLLM 0.23.1 26.8/30.0/31.3; its only vLLM 0.25.0 point is cache-off c32 at 30.6 ([results](https://github.com/Weschera/qwen-sglang-dgx-spark/blob/03253ef98c01de59a21c85b9a5cc6a27a871c383/results/results.csv)). | Consistent with a configuration issue, but not project-grade evidence: old/mixed vLLM versions, one scalar throughput field, and incomplete repetitions/requests. |
| A residual cache-on SGLang lead is plausible but unproven. | The CSV reports 35B-only SGLang/vLLM 0.23.1 values of 324.4/261.6 at 64k/c32, 85.3/63.8 at 256k/c2, and 133.8/92.6 at 256k/c8; the README says these cells have only one or two runs ([README lines 61-72](https://github.com/Weschera/qwen-sglang-dgx-spark/blob/03253ef98c01de59a21c85b9a5cc6a27a871c383/README.md#L61-L72)). | Candidate to reproduce, never a denominator. There is no vLLM 0.25 cache-on result in the repository. |
| The launch policies are not equivalent. | The checked-in vLLM arm forces FP8 KV cache and memory fraction 0.7 while the SGLang arm leaves KV dtype at `auto` and uses 0.75 ([vLLM lines 19-24](https://github.com/Weschera/qwen-sglang-dgx-spark/blob/03253ef98c01de59a21c85b9a5cc6a27a871c383/serve-vllm.sh#L19-L24), [SGLang lines 18-24](https://github.com/Weschera/qwen-sglang-dgx-spark/blob/03253ef98c01de59a21c85b9a5cc6a27a871c383/serve-sglang.sh#L18-L24)). SGLang `auto` means model dtype at `server_args.py:558-570` in `f63458b`; this checkpoint has no FP8 KV declaration. Both arms also enable three-token MTP using different frontends. | First isolate cache implementation with BF16 KV and speculative decoding off. Equal byte capacity replaces unequal memory fractions. Add a later composed MTP gate; never mix it into the cache attribution run. |
| vLLM does support Qwen hybrid prefix caching, but not by default. | vLLM v0.25.0 `vllm/config/model.py:1845-1900` returns unsupported for a hybrid model's default policy. An explicit enable resolves `mamba_cache_mode=align`, block-aligned state retention, and required chunked prefill at `vllm/model_executor/models/config.py:558-602`; Qwen3.5 rejects `all` and names `align` at `vllm/model_executor/models/qwen3_5.py:294-301`. | The matched vLLM arm must say `--enable-prefix-caching --mamba-cache-mode align`; relying on a generic dataclass default is invalid. |
| SGLang has a structurally different hybrid-cache path. | SGLang v0.5.15 chooses `MambaRadixCache` for hybrid SSM models at `python/sglang/srt/mem_cache/registry.py:78-131`; its cache owns separate full/Mamba LRU state and prefix matching at `mamba_radix_cache.py:424-523`. | This is a credible source-level reason a residual gap could be real. Only matched hit proof, timing, memory, and paired traces decide whether it is faster on GB10. |
| The external harness is insufficient for this project's gate. | `spark-bench` tier2 sends the same prompt to each stream; the published CSV exposes aggregate output throughput, one TTFT percentile, completion count, and notes. It does not prove token-ID correctness, cache-hit length, full TTFT/TPOT/ITL/E2E percentiles, request/s, peak memory, no eviction, three clean repetitions, or paired kernel traces. | Retain it as a reproduction lead only. The project harness must emit the complete evidence contract below. |

### Two workload classes, never one blended score

| Class | Cache policy | Corpus | Purpose |
|---|---|---|---|
| Cache-neutral | Explicitly off on all three arms: SGLang `--disable-radix-cache`; vLLM/vllm.cpp `--no-enable-prefix-caching` | Existing exact 1024/128 disjoint corpus; pairwise common prefix no more than 32 tokens | Scheduler/kernel/frontend comparison without cache hits. Remains owned by `BACKEND-GATE-CUDA-SGLANG`. |
| Shared-prefix | Explicitly on: SGLang Mamba radix; vLLM `align`; vllm.cpp `KV-MAMBA-ALIGN` | Exact common prefix followed by disjoint per-request suffix; separate 64k and 256k shapes | Cache lookup, GDN/Mamba-state retention, shared physical blocks, branch admission, and hit-path latency. Owned only by `BACKEND-GATE-CUDA-SGLANG-PREFIX`. |

Every report presents both classes side by side. A cache-on result never closes
cache-neutral parity, and cache-neutral parity never claims cache-on support.
The faster equivalent reference is selected independently for every metric and
workload point.

### Canonical shared-prefix corpus

| Shape | Exact common prefix | Exact unique suffix | Exact input/output | Concurrency | Repetitions |
|---|---:|---:|---:|---|---:|
| 64k | 63,488 IDs | 1,024 IDs | 64,512 / 128 | 2, 8, 32 | 3 independent reset/seed/run repetitions |
| 256k | 260,096 IDs | 1,920 IDs | 262,016 / 128 | 2, 8 | 3 independent reset/seed/run repetitions |

The project tokenizer generates and verifies the common segment and every
suffix; all three engine tokenizers must reproduce the stored IDs exactly. The
suffixes are pairwise disjoint from one another and from warmup. Output is
greedy, `ignore_eos=true`, exactly 128 native IDs, and speculative decoding is
off. Each repetition performs this sequence under one server lifetime and one
GPU lock:

1. reset the engine cache and counters and prove zero starting hits;
2. issue one untimed seed request containing the exact common prefix, wait for
   its insertion, and record cold-fill latency and memory separately;
3. snapshot cache counters/capacity, then issue exactly one simultaneous timed
   wave of the stored unique branches at the target concurrency;
4. prove every request hit the expected engine-specific aligned common length,
   no block/state was evicted, and only each suffix plus the mandatory final
   common-prefix token was computed;
5. retain native output IDs, raw streaming events, cache-counter deltas, memory
   samples, server logs, and thermal state; then reset before the next point.

The 256k shape is mandatory after 64k closes, not a substitute for it. It may be
marked `PENDING hardware capacity` only with the measured capacity calculation
and next command; an OOM, timeout, or eviction is a failed/void run, never a
smaller silently substituted context.

### Matched server policy

- All arms use BF16 KV/SSM storage for the first binding cache gate. SGLang must
  receive `--kv-cache-dtype bf16`; vLLM receives `--kv-cache-dtype auto` only
  after startup proves BF16, otherwise use its explicit BF16 spelling. Ours
  records the exact native dtype. A later FP8-KV matrix row may add a separate
  equal-FP8 gate.
- Speculative decoding, tool/reasoning parsing, LoRA, multimodal work, and
  external KV connectors are disabled. After the isolated cache gate closes,
  repeat a composed MTP-on production gate under the speculative-decoding row.
- Chunked prefill is enabled and maximum batched tokens is exactly 8192 on all
  arms. The reported recipe warns that 65,536 hung the GB10; no campaign may
  probe that value ([README lines 99-101](https://github.com/Weschera/qwen-sglang-dgx-spark/blob/03253ef98c01de59a21c85b9a5cc6a27a871c383/README.md#L99-L101)).
- Usable cache capacity is matched in bytes and usable token/state slots, not
  by nominal memory-fraction flags. Record page/chunk sizes, full-attention KV
  bytes, GDN state/checkpoint bytes, reserved graph/workspace bytes, and the
  maximum no-eviction branch count before timing.
- Production CUDA graphs remain enabled. The same immutable checkpoint,
  language-only path, quantization semantics, tokenizer, seed, output length,
  request order, and host client drive every arm.

### Hit proof, metrics, and traces

vLLM evidence includes the deltas of `vllm:prefix_cache_queries` and
`vllm:prefix_cache_hits` (`vllm/v1/metrics/loggers.py:547-564,1088-1092` at
`702f481`) plus per-request cached-token counts and the resolved `align` startup
log. SGLang evidence includes the selected `MambaRadixCache`, per-request matched
prefix length from scheduler/cache instrumentation, full/Mamba protected and
evictable sizes before/after, and an explicit zero-eviction assertion. vllm.cpp
must expose equivalent native counters under `KV-MAMBA-ALIGN`; log inference or
faster TTFT alone is not hit proof.

For every point, retain request/s; input, output, and total tok/s; mean and
P50/P90/P99 E2E, TTFT, TPOT and ITL plus P95/max ITL; cold-fill time; exact hit
length/rate; completed/errors/concurrency; process-tree PSS/RSS; host
`MemAvailable` delta; numeric accelerator memory when available; and three-run
spread/CV. Correctness remains 16/16 engine-native output IDs against the fresh
vLLM oracle before SGLang can bind. A cache hit is never allowed to trade away
token equality.

Capture warmup-excluded system-wide CUDA traces for all three engines at
64k/c32 and 256k/c8, with the seed/fill outside the timed NVTX range. Diff
kernel names, launches, GPU time, computed-token counts, cache/state copies,
and host gaps. Trace names are ground truth for the actual selected runtime
path; source similarity is not.

### Tests to port/add for the prefix row

| Source | Local requirement |
|---|---|
| vLLM `tests/v1/e2e/general/test_mamba_prefix_cache.py` at `702f481`, including repeated/shared-prefix hit checks around lines 755-757, 991-993 and 1042-1047 | Port exact Qwen hybrid align hit/miss, block-boundary, branching, no-eviction, output-equivalence, reset, and counter tests into KV-manager and gate-model tiers. Unsupported tests land skipped with a tracked row reason. |
| SGLang v0.5.15 `MambaRadixCache.match_prefix`, insertion, finished-request caching, split/eviction and Mamba/full LRU tests | Re-express the behavioral cases against `KV-MAMBA-ALIGN`; do not copy the Python data structure as a new incompatible cache abstraction. |
| `Weschera/spark-bench` identical-prefix tier2 behavior | Corpus contract test proves byte-stable common IDs, suffix divergence, exact lengths, simultaneous admission, reset/seed order, and that cache-off reports zero hits while cache-on reports the expected aligned hit. |
| Project benchmark protocol | Synthetic raw artifacts exercise every metric, faster-per-axis floor selection, missing-hit/eviction/dtype/spec mismatch voiding, three-repetition enforcement, and 64k/256k separation. |

### Prefix-row work breakdown and gate

| Order | Deliverable | Exit condition |
|---|---|---|
| PX1, CPU/read-only | Repin the competitor wrapper to SGLang v0.5.15 `f63458b`, add deterministic long-prefix corpus/manifests, cache counters, equivalence validation, and synthetic tests | No GPU; exact image/source/model/client hashes and all fail-closed contracts pass. |
| PX2, engine | Write `specs/mamba-align-retention.md`, then implement `KV-MAMBA-ALIGN` plus native hit/eviction/capacity counters and ported tests | Unit/integration tests prove aligned Qwen GDN/full-attention retention, branching, reset, no eviction, and output identity. |
| PX3, GPU preflight | Exact 27B load/token/dtype/capacity/hit proof for all three arms; SGLang v0.5.15 image platform digest verified | Three equivalent arms or an explicit non-equivalence/blocker; no performance claim. |
| PX4, 27B gate | Three repetitions of all 64k then 256k points and paired traces under one uninterrupted series lock | Ours meets or beats the faster equivalent vLLM/SGLang result on every throughput, latency, and memory axis; exact outputs/hits/no-eviction are preconditions. |
| PX5, 35B gate | Run only after every applicable 27B cache-off and cache-on axis passes and exact 35B SGLang quant equivalence is proven | Same every-axis rule for 35B; otherwise the row remains open with the exact failed axis. |
| PX6, composed production | MTP enabled equivalently after the isolated cache and speculative-decoding rows close | Separate report attributes composition; it cannot retroactively replace PX4/PX5. |

`BACKEND-GATE-CUDA-SGLANG-PREFIX` closes only when both models' applicable
64k/256k points satisfy correctness, hit/no-eviction, every-axis performance,
memory, trace, and reproduction gates. Until then its state may advance through
implementation/gating, but no external scalar is published as an accepted
project result.

## P1 CPU checkpoint (2026-07-11)

`BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` P1 is implemented and handed to
`GATING`, not `DONE`. The deterministic generator writes the exact SGLang
two-turn custom-dataset shape plus stored token IDs/hashes, refuses non-empty
artifact roots, validates the complete corpus before exposing partitions, and
enforces global prompt disjointness plus the 32-token common-prefix ceiling.
The client wrapper constructs only the digest-pinned `--pull=never` command,
runs native `output_ids`, non-streaming usage, and incremental SSE preconditions,
validates all raw detail arrays and achieved concurrency, and refuses an
existing output file. The summary recomputes TTFT/ITL percentiles, repetition
spread/CV and direction-aware floor ratios while propagating checkpoint,
quantization, tokenizer, native-ID, usage and streaming precondition failures.
The 100 ms sampler unions ordinary descendants with every PID in an explicitly
owned cgroup tree, then aggregates RSS/PSS and host `MemAvailable`; numeric GPU
memory is retained and `N/A` stays JSON `null`.

Source inspection corrected one assumption in the accepted spike: pinned
SGLang `--output-details` emits `input_lens`, `output_lens`, `ttfts`, nested
`itls`, texts and errors, but **not per-request E2E latencies**. It reports only
aggregate E2E/TPOT fields and has no P90 TPOT field. P1 therefore retains those
aggregates, emits raw P90 TPOT as unavailable, and makes the group non-binding;
it does not approximate end-of-response time from last-token time. P2 must
capture the missing raw values without modifying timed request/metric semantics
before a binding every-percentile result can be promoted.

CPU evidence: direct unittest discovery passes **16/16**; the same suite passes
through the new CMake/CTest registration **1/1**; `py_compile`, `bash -n`,
ShellCheck and the real shell dry-run pass. The dry run records one planned
whole-campaign lock, `--pull=never` timed clients, provisioning outside that
lock, and every pending evidence precondition. No image was pulled, no SGLang
package installed, no checkpoint loaded or mutated, and no GPU/performance
command ran. P2 exact 27B/35B load/quant/token classification and P3 binding
campaign remain open.

## Risks and fixed decisions

| Risk | Fixed handling |
|---|---|
| 35B mixed ModelOpt metadata selects SGLang `w4afp8` instead of the checkpoint's per-layer FP8/NVFP4 path | Treat as unresolved until exact tensor load and token evidence. Never force a convenient flag and call it equivalent. |
| 27B format name matches but global-scale convention or packed layout differs | Audit real tensors/scales and trace selected kernel before binding. |
| Our buffered SSE makes TTFT/ITL look anomalous | Hard-fail the streaming validity probe; `SERVE-ASYNC-LLM` is a dependency, not an excuse attached to a number. |
| Prefix cache or repeated warmup prompts improve one arm | Disjoint 1024-token partitions, bounded shared prefix, dedicated warmup set, production cache left enabled. |
| SGLang client counts requested output length when streaming usage is absent | Independent native-output-ID plus non-stream usage preflights; retain exact IDs and raw chunks; void on mismatch. |
| Official image tag moves | Digest pin plus OCI revision and platform-digest record; `--pull=never` during execution. |
| Container provisioning pollutes a timed run | Pull/decompress outside the GPU campaign and outside any active benchmark window; one later execution lock covers all arms. |
| GB10 `nvidia-smi` reports unified memory as `N/A` | Common process-tree PSS and system available-memory sampler; retain numeric accelerator data only when actually available. |
| Graph/JIT warmup contaminates profiles | Warm all concurrency points first; system-wide nsys starts afterward on disjoint trace prompts. |
| HTTP/client implementation differs | Same pinned SGLang OpenAI completions function drives every endpoint; backend name only selects URL semantics. |
| A faster result uses different output or checkpoint | Correctness, length, corpus, model, and quantization checks are preconditions. The result is otherwise void. |

There is no remaining product choice in this spike. The implemented preflight
row `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` is `GATING`; the cache-neutral row
`BACKEND-GATE-CUDA-SGLANG` remains `BLOCKED` until its two dependencies close.
The new `BACKEND-GATE-CUDA-SGLANG-PREFIX` row is `READY` for PX1 work; PX2
starts by writing the dedicated `KV-MAMBA-ALIGN` leaf spike. Its binding
performance run remains dependency-blocked as specified above. The
35B SGLang floor remains conditional on exact-artifact compatibility evidence.
