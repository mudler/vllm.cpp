# Matched Qwen3-4B BF16 benchmark on Strix Halo

Row: `BACKEND-GATE-ROCM-SGLANG`. Issue: [#3053](https://github.com/mudler/vllm.cpp/issues/3053).

## Now

Preparation is pending. The developer approved this scope on 8 September 2026.
Source registration is not a successful build or model run. No speed claim exists.
This leaf does not replace the 27B obligations in #3043 or the quantized token gate.

## Scope and pins

Run vllm.cpp, vLLM, SGLang, and llama.cpp on the same leased `strix:gpu0`.
Use `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`.
The staged source is `/workspace/ckpt/qwen3-4b-bf16`; its HF download metadata
names that revision. Verify every input file against an operator manifest before use.
The three safetensors shards total 8,044,982,000 bytes. Never read model weights
from the shared filesystem during timed work. Copy to a unique local directory.

Engine source pins:

- vllm.cpp: `6e3cbfb940be89e28d1d71c264fd8c3a4e44afeb`.
- vLLM: `e126687a9a828d513c01a07cd69f025f27d63280`.
- SGLang: `f63458b5beaceabbd9d749b9fc956370e1b649e6`.
- llama.cpp: `10bf611e533d81f739128304991c5e133c6aebd8`, stock b10451.

The task branch starts at the vllm.cpp pin and depends on PR #3052's verified
oracle preparation. Keep the old 27B worker and its evidence unchanged.
One pull request carries this spec, preparation, tests, and resulting evidence.
No model, engine, or oracle pin changes silently after a failed result.

## Source anchors

- vLLM `vllm/model_executor/models/qwen3.py:65,105,114,271` supplies dense Qwen3.
- SGLang `python/sglang/srt/models/qwen3.py:44,64,453,719` implements and registers Qwen3.
- SGLang `docker/rocm.Dockerfile:24,80,285-300` supplies the ROCm package recipe.
  Its named stages target gfx942/gfx950, not gfx1151. A CUDA wheel or changing
  only the stage argument does not establish a Strix build.
- llama.cpp `src/models/qwen3.cpp:8` recognizes the dense 4B model.
  `conversion/qwen.py:155` registers conversion. `convert_hf_to_gguf.py:60,211`
  supports BF16; `conversion/base.py:934-936,998-999` promotes one-dimensional
  tensors to F32 and otherwise selects BF16. Promotion preserves values but
  does not make the containers byte-identical.
- Ours `src/vllm/model_executor/models/qwen3_dense.cpp:62-75` loads safetensors;
  its GGUF arm is refused. Use the native safetensors path.

Read the executing build and runtime chain before selecting dependencies or flags.
Use supported production attention and graph paths, not an eager vLLM denominator.
Record architecture targets of every compiled extension and the resolved backends.

## Matched workload

Use the six raw text prompts from `tools/bench/strix_vllm_oracle/runtime.py`.
Re-tokenize with Qwen3-4B's own tokenizer; do not reuse the 27B token IDs.
Require identical prompt IDs on every engine. No chat template or reasoning wrapper.
Temperature 0, top-p 1, no speculative decoding, ignore EOS, exactly 128 output tokens
per request. BF16 model dtype and BF16 KV cache on all arms; record actual resolution.
llama.cpp consumes a BF16 GGUF converted from this exact safetensors checkpoint.
Audit mapped tensor shapes and numerical values, including tied embeddings and
F32 promotions. Refuse missing, duplicated, changed, or unexplained extra tensors.

Run concurrency 1 and 4 with context capacity 2048 per sequence and at least four
request slots. Use one identical ordered six-request corpus at each concurrency.
Disable reusable prefix/prompt caching consistently so repetitions execute the same
prefill work. Keep model weights resident throughout warm-up and timed repetitions.
Warm up once per workload configuration, then retain all three measured repetitions.
Do not select the fastest repetition. Record per-request timings and token IDs.
Use a common timing boundary, excluding model loading, graph compilation, and warm-up.
Report output tokens divided by elapsed workload wall time, not an average of
per-request token rates. Report median and range across repetitions, TTFT, inter-token
latency where measured, total latency, and peak host/GPU memory with method stated.
Do not manufacture per-token latency from a non-streaming response.

Run the four engines serially in one lease. Stop each server and verify process
teardown before starting the next. Record memory headroom, device identity, contention,
clock samples, exact commands, source and binary hashes, resolved settings, and errors.
Use bounded subprocesses, bounded logs, build parallelism at most four, and ccache.
Keep Python dependencies isolated. No GPU resets, driver changes, or lease bypass.

## Correctness and acceptance

First qualify all four model loads and complete outputs. Compare the greedy token
sequences against the pinned vLLM results for this workload, including repeatability.
A mismatch is a failing token-exact gate, not permission to introduce a tolerance.
Distributional acceptance requires an explicitly ratified gate and measured oracle
nondeterminism. Do not infer that gate from another model's record.
Keep diagnostic timings separately labeled if correctness fails; do not accept or
publish a speed ratio from a failing arm. Every requested engine gets a result or a
precise pending/failing reason. No promised winner and no cross-model substitution.

## Task 1: prepare the SGLang Strix environment {id: 1, deps: []}

Files: new `tools/bench/strix_four_engine/prepare.py` and focused tests in
`tests/tools/test_strix_four_engine.py`; scoped preparation evidence in this spec.
Interface: `prepare.py --manifest PATH --output UNUSED_DIRECTORY` inside a Strix
lease. The manifest binds the SGLang source archive/revision/hash, explicit dependency
artifacts or requirement pins, and resource limits. Output includes provenance,
commands, logs, installed package and extension identity, and success/failure state.
Use the pinned upstream ROCm build entry point with explicit gfx1151 compilation.
Inspect the upstream source for actual build variables, prerequisites, and imports.
Refuse a CUDA-only dependency stack, wrong device, unexplained source mutation, unbound artifacts,
architecture mismatch, timeout, low headroom, or overwrite. Do not install into the
previous vLLM venv. Do not bypass a dependency conflict with an unverified mixed stack.
No model generation or throughput claim belongs to this task.

Test-first: exercise the real CLI with bounded fake external commands and explicit
fixture manifests. Assert the selected ROCm build path, gfx1151 target, isolated venv,
provenance output, failures, exit status, and no model or GPU work outside the lease.
Mutate every claimed refusal and the production CLI call sites in independent review.
Verify: `python3 -m unittest tests.tools.test_strix_four_engine`.
Run the repository preflight and report argument-dependent skips separately.
The operator runs the reviewed preparation under a lease and retains real evidence.

## Task 2: convert and qualify the model {id: 2, deps: [1]}

Use the exact checkpoint and stock pinned llama.cpp converter. Add the smallest
reviewed tensor audit and CLI load/ID capture adapters needed for all four engines.
Prefer existing production entry points and upstream tests over private model calls.
Write a committed task supplement with exact adapter interfaces and focused Verify
command before implementation, informed by Task 1's actual environment. Do not hide
unresolved interfaces behind guesses. Conversion audit and all four loads must pass
before the throughput gate runs.

## Task 3: execute and report matched measurements {id: 3, deps: [2]}

Reuse a suitable pinned upstream serving client where possible. Commit the precise
client adaptation and tests before use. Preserve all warm-up and measured runs with
their disposition. Fresh implementation, independent mutation review, and operator
verification apply to every code change. Update this spec with measured outcomes and
remaining gaps. A public benchmark detail and index row are added only if acceptance
permits publication. Never turn four engine names into a four-way result without runs.

## Risks and stop conditions

SGLang's AMD dependencies may not support gfx1151 at the pinned source. BF16 kernels
or KV handling may be unsupported in an engine's actual HIP path. Conversion can
alter tensor layout or tokenization. Short prompts expose scheduler overhead and are
not evidence about 27B throughput. Report these separately from the earlier campaign.
Stop for a necessary oracle-pin change, a changed model
or precision, hardware instability, or scope beyond preparation and measurement.
File an in-flow bug when a small scoped repair is needed; independently review it.
Do not port a new engine backend or model silently to make the benchmark runnable.
The explicitly approved gfx1151 compatibility patch below is within scope.

## Approved patched-SGLang preparation

On 8 September 2026 the developer explicitly approved a tracked gfx1151 compatibility
patch and required the comparator label **patched SGLang**. Keep the upstream source
pin unchanged and record the patch digest separately. This is not stock SGLang evidence.
Do not change the global SGLang oracle pin or its existing gateability record.

The unmodified `sgl-kernel/setup_rocm.py:77-81` rejects gfx1151. GPU detection
overrides the environment target. The alternative AMD wheel CMake path also rejects
it. Qwen3 imports Qwen2MLP, which imports activation; HIP activation imports the compiled
`sgl_kernel.common_ops` extension. Triton attention does not eliminate this dependency.
CPU-only execution with a gfx1151 property fixture reproduces exit 1 before compilation.

Task 1 additionally owns a versioned patch under `tools/bench/strix_four_engine/patches/`
against the exact SGLang source pin. Apply only after source archive verification, with
strict patch context and no fuzz, then assert the complete expected changed-file set.
Preserve upstream-before, patch, and patched-after hashes. The patch may admit gfx1151,
select the correct FP8 encoding, constrain shared-memory requests to the device limit,
and repair wave-size or HIP compilation assumptions encountered in this build. Every
change must name its hardware reason and preserve gfx942/gfx950 behavior.
Do not merely remove the target guard or lie about the architecture. Unknown targets
must remain refused. Do not stub a required operation, drop production call sites,
replace GPU work with CPU fallback, or disable a failing numerical test.

Test the real upstream setup policy before and after patching with CPU fixtures for
gfx1151, gfx942, gfx950, and an unknown target. Prove deterministic patch application,
unchanged unrelated source, architecture detection, FP8 and LDS selection, and explicit
patched-comparator labeling. Port applicable upstream numerical tests for each affected
kernel; execute those tests in the Strix lease before model qualification. Record a
kernel that is compiled but unreachable for this BF16 dense workload without claiming
it is validated. A new required algorithm or unsupported hardware instruction that
cannot be repaired within this compatibility scope remains a named blocker.

Task 1 success requires the patched extension to build for gfx1151, import in its
isolated ROCm environment, and pass the declared affected-kernel tests. The operator
then proceeds to conversion, correctness, and the approved c1/c4 throughput measurements.

## Task 1 preparation implementation evidence

The adapter accepts a flat git archive with the pinned revision in its PAX
comment. The operator manifest binds the archive and every dependency wheel by
SHA256. It supplies an absolute Python interpreter, expected Torch and Triton
versions, and explicit resource limits. The adapter creates its own local
environment and installs only the bound wheels. Offline HIP-extra resolution
and `pip check` must pass. CUDA identity and overlapping Triton distributions
remain refusals. No previous environment is modified.

The versioned patch changes only `sgl-kernel/setup_rocm.py` at `f63458b5`.
The complete setup hash changes from `8a37e537157e76cd2064e971842b9f7f7c8d075942b980f478c945afd876d6ac`
to `5fe84d7afb969073bff8b1bfcb0e224c36964aca0e7b025ea25caec55407fe9a`.
The patch admits gfx1151, selects E4M3 FN, and limits dynamic TopK LDS to 48 KiB.
The operator measured 65536 bytes of shared memory per block. The remaining
16 KiB accommodates static allocations. gfx942 retains FNUZ and 48 KiB.
gfx950 retains FN and 128 KiB. Unknown targets remain refused. No wave-size
or instruction change is claimed without a compiler failure that requires it.

The adapter preserves upstream activation and TopK test files and their cases.
Its additional BF16 test uses the upstream SiLU reference expression at Qwen3-4B
width 9728 and decode concurrency 1 and 4. The added dtype uses explicit
PyTorch BF16-default tolerances, `rtol=1.6e-2` and `atol=1e-5`.
The original FP16 tolerances remain unchanged. A CPU spy test checks that the
added test enters the compiled operation and calls its numerical assertion.
This is wiring evidence, not a BF16 numerical result. All numerical tests must
execute under the lease without failures or skips. Multi-device all-reduce
kernels are compiled but not validated by this single-device BF16 workload.

The selected AMD Torch `2.9.1+rocm7.2.1.lw.gitff65f5bc` wheel's
`torch/utils/cpp_extension.py:2842` selects `PYTORCH_NVCC` before the HIP compiler default. The adapter sets
`ccache hipcc`, requires that generated Ninja rule, and records its hash.
It records the patch, source hashes, wheel hashes, runtime identity, extension
targets, package versions, bounded command logs, and the final disposition.
The comparator label is always `patched SGLang`.

The first `python3 -m unittest tests.tools.test_strix_four_engine` invocation
failed because the preparation entry point did not exist. Evidence:
`/dev/shm/strix3053-cli-red.log`. The exact upstream setup policy also exited 1
for gfx1151 before the patch. Dedicated ccache and BF16-call-site regressions
were red before those paths were implemented. Their logs are
`/dev/shm/strix3053-ccache-red.log` and `/dev/shm/strix3053-bf16-red.log`.
The focused suite passed five tests with parameterized failure cases in
`/dev/shm/strix3053-focused-final.log`. Scratch mutation results and the final
immutable-head preflight are reported in the implementer handoff.

The real patched build, imports, affected-kernel numerical tests, and model
qualification remain the operator's lease obligations. CPU fixtures establish
no hardware support or benchmark acceptance.

## Task 1 runtime-loader repair contract

The required manifest `rocm` object contains an absolute `root`, a `compiler`
record, and a nonempty `runtime` list. Each file record contains an absolute
`path` and `sha256`. The compiler path names executable `hipcc`.
Resolve the root and every file before use. Refuse missing files, incorrect
hashes, relative paths, duplicate resolved file bindings, and files outside
the resolved root. Toolkit symlinks are valid only when their targets stay
inside that root. Require an existing compiler directory and runtime directory.

Construct `PATH` from existing `root/bin` and `root/llvm/bin`, followed by
the fixed `/usr/bin:/bin` system-tool directories. Construct `LD_LIBRARY_PATH`
only from existing `root/lib` and `root/lib64`. Refuse directory targets outside
the root and a selected `hipcc` different from the bound compiler. Ignore
inherited loader paths and replace toolkit-selection variables with the selected
root. Preserve lease device visibility and the existing compiler-override refusals.
Do not change the system loader configuration or install global dependencies.

Before commands, record the root, resolved file identities, and constructed
paths. Record `root/llvm/bin/clang++` identity when present. Revalidate the bound
files and selected paths after preparation, before publishing success.
The real CLI tests cover selection, refusals, and post-build changes using CPU
fixtures. Verify with `python3 -m unittest tests.tools.test_strix_four_engine`.
This repair establishes no GPU numerical result. The operator retains the
real build and hardware gates.

The selected-path CLI regression exited 1 before implementation because `hipcc`
was not found. Evidence: `/dev/shm/strix3053-loader-red-selection.log`.
The repaired focused suite passed 8 tests, including contained symlinks and
19 invalid-binding cases, in `/dev/shm/strix3053-loader-green-final.log`.
The code does not change the shared ROCm installation. Hardware verification
and the independent review remain separate obligations.

## Task 1 review repairs

The emitted identity probe now recognizes `pytorch-triton-rocm` as a Triton
namespace owner. Two installed owners remain a refusal. The regression executes
the emitted code against distribution fixtures instead of supplying prepared JSON.
The BF16 wiring test now checks the expected operand's SiLU expression and slices.
Patch tests independently verify the digest and complete before/after inventories.
Postpatch fixtures corrupt the setup or add, change, and delete another source file.
Toolkit tests cover same-content symlink escapes and optional Clang provenance.

The selected Torch wheel invokes bare `ninja` at `cpp_extension.py:2295,2564`.
The adapter records the actual build-PATH executable, resolved path, hash, and version.
It refuses a changed executable before success. This record does not claim that
the Ninja wheel installed in the environment supplied the executed binary.
The fixed toolkit PATH remains unchanged.

The alias regression first failed because overlapping distributions were accepted.
The Ninja regression first failed because provenance was absent and mutation was accepted.
Evidence is in `/tmp/strix3053-review-fixes-red.log` and
`/tmp/strix3053-review-fixes-ninja-red.log`. The repaired focused suite passed
10 tests in `/tmp/strix3053-review-fixes-green.log`.
Scratch mutations remove each reviewed guarantee and must fail the focused tests.
These CPU checks do not establish GPU support or benchmark correctness.

The Ninja CLI regression independently changes the selected PATH entry and
the resolved executable target. Both replacements retain executable mode and
identical bytes. The PATH case retains the original resolved target. The target
case retains the original selected path. Each missing comparison therefore
fails its own case without relying on a changed digest.
The focused suite passes 11 tests. Scratch deletion of each comparison causes
the corresponding case to accept preparation incorrectly and fail its assertion.
Evidence logs are `/home/mudler/.cache/strix3053-ninja-green.log`,
`/home/mudler/.cache/strix3053-ninja-red-selected.log`, and
`/home/mudler/.cache/strix3053-ninja-red-resolved.log`.
The final repository preflight is recorded in
`/home/mudler/.cache/strix3053-ninja-full.log`.

## Owed

- #3053 owns real SGLang gfx1151 preparation, verified conversion, all four loads,
  correctness, and matched measurements. All remain pending at spec creation.
- #3043 retains the prior Qwen3.8-27B obligations; this smaller-model benchmark
  establishes no speed advantage for that model.
