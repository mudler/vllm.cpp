# Current-pin vLLM on Strix Halo

Row: `BACKEND-GATE-ROCM-VLLM`
Issue: [#3043](https://github.com/mudler/vllm.cpp/issues/3043)
In-flow runtime repair: [#3048](https://github.com/mudler/vllm.cpp/issues/3048)
Base: `e2fb2f06d9944c4bbe66531034479d370df67815`

## Now

The developer approved the sequence on 7 September 2026: current-pin
baseline, packed decode, then GGUF optimization. This change establishes
the first prerequisite. It does not implement the inference optimization.
The isolated current-pin vLLM and GGUF plugin build on Strix is complete.
Its build state records `BUILT` after dependency, source, compiler, and
extension checks. Six-prompt generation and eight upstream packed-test cases
remain owed. The first generation attempt failed at the metadata RPC under
#3048 before token generation. Model gateability remains PENDING. The row
lifecycle is unchanged.

## Scope and exclusions

Build vLLM at the active repository pin in an isolated environment on
`strix:gpu0`. Build its pinned GGUF plugin against the same environment.
Run Qwen3.8-27B Q4_K_M in production compilation mode. Preserve the old
oracle, all existing environments, source checkouts, models, and captures.
Capture the current upstream packed-decode tests as the next port's oracle.

Do not advance the global pin, alter product code, replace the old oracle,
change graph admission, or change quantization defaults. Do not publish an
accepted performance ratio. `TOKEN_GATE=FAIL` is carried from the survey,
not repaired by successful generation. A failure stays visible and nonzero.

Use one issue change with this spec committed before its harness. Repository
policy defaults to one integration change. The campaign's recorded local
merge authority applies after fresh review and operator verification.

## Source and runtime evidence

The active vLLM pin is `e126687a9a828d513c01a07cd69f025f27d63280`.
Read `.agents/upstream-sync.md` for its identity contract. Historical
`5559679229` measurements cannot stand for this pin.

The GGUF plugin pin is `d4c1f0d082fc7cd4350da56689109a01c1f29d6c`.
Its archive SHA256 is
`9e15c20e0b75f75bbf886966df07843c4b70a7952fad4b80e8e8183e2f70743b`.
The archive is retained beside the previous build recipe under
`docs/bench-evidence/oracle-vllm-gfx1151-20260903/` by reference to its
captured shared artifact. New manifests supply paths, never infer them.

Read-only lease `d24f3a86-70fc-4d77-b4c1-54deb998715b` found installed
vLLM distribution `0.26.0.dev0+g5559679229.rocm724`, Torch
`2.13.0+rocm7.2`, Triton `3.8.0`, and plugin `0.0.5`.
Lease `eab2c163-9fe2-44a5-b886-1dc33f9c9224` found native plugin
MMVQ predicates enabled for Q4_K, Q5_K, Q6_K, and Q8_0. Its extension hash
matches the historical capture. Neither probe ran the model or proved a
current-pin dispatch. The old installation is not a build cache to overwrite.

Pinned upstream anchors:

- `vllm/plugins/__init__.py::load_general_plugins` loads the GGUF plugin.
- Plugin `quantization/linear.py::_fused_mul_mat_gguf` selects MMVQ at
  one token for these quant types. `ops.py::ggml_mul_mat_vec_a8` chooses
  the installed native extension or Triton fallback.
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py::_forward_core_decode_non_spec`
  calls the packed recurrence after convolution.
- `vllm/third_party/flash_linear_attention/ops/fused_recurrent.py::fused_recurrent_gated_delta_rule_packed_decode_kernel`
  retains normalized Q/K and beta in FP32 registers.
- `tests/kernels/test_fused_recurrent_packed_decode.py::test_packed_decode_keeps_beta_in_fp32`
  is the smallest current-pin numerical regression. `git log -S` identifies
  upstream `56058fd572`, issue vllm#53877, as its introduction.

## Harness design

Add a bounded worker and CPU tests under `tools/bench/strix_vllm_oracle/`
and `tests/tools/`. The operator supplies a JSON manifest and an unused
output directory. Preserve a normalized copy and SHA256 of the manifest.
The worker requires `RC_DEVICE=strix:gpu0` and a nonempty `RC_JOB_ID`.
Reject architecture overrides and inherited experimental tuning. Do not
silently reuse an output directory or a success marker from another run.

The controller supplies `CUDA_VISIBLE_DEVICES=0` on this named lease.
The operator measured this value in job `33bb569e-d273-4268-a1ea-90445c5cf0c4`.
Accept exactly this inherited visibility value after validating the lease.
Preserve it in every child and record it in `environment.json` for both phases.
Reject every other supplied value and every other inherited `CUDA_` setting.
The existing `HSA_`, `HIP_`, and other experimental-setting refusals remain.
An absent visibility variable remains accepted for existing controlled fixtures.

The manifest binds each source archive's revision, SHA256, and path. Verify
the vLLM archive's git-archive commit marker as well as its hash. Verify the
plugin archive against its recorded hash. Extract without path traversal or
escaping links into newly created local storage. Record the source digest
before and after building, distinguishing generated build outputs from source.

Create a new virtual environment and build directory. Builds use local `/tmp`
because shared CIFS storage cannot carry build symlinks. Export retained
artifacts with dereferenced copies. Dependencies stay isolated. An explicitly
supplied old environment may supply a package cache or read-only dependency
inventory, never an installation target or silently imported old vLLM.
Use the ROCm Torch package, ccache, `MAX_JOBS=4`, and only `gfx1151`.
Record compiler, ROCm, Torch, Triton, installed distributions, build commands,
logs, source archives, wheel hashes, extension hashes, and device code targets.
Do not relabel an old binary using a version string. Current source must build.

Separate build and run phases. A run consumes successful build state, verifies
the same manifest and installed source/extension identity, and refuses stale
state. Import vLLM from outside its source directory. Assert the resolved
platform is ROCm and the reported device is genuinely gfx1151. Record the
runtime version and distribution version, allowing a measured ROCm packaging
suffix without substituting a different commit.

The run copies the model and its required configuration/tokenizer assets to
unique local storage. Verify the local model before loading:
`Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 bytes, SHA256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
The operator manifest binds the configuration, tokenizer, and required mmproj
assets by content. Reuse already staged assets rather than download new models.

Reuse the six explicit prompt-ID lists in the historical `gen_rocm.py`.
Assert tokenization agrees. Generate 48 tokens for every prompt with
temperature zero, top_p one, ignore_eos true, and max_num_seqs one.
Use max_model_len and max_num_batched_tokens 2048, memory utilization 0.60,
image/video limits zero, GGUF quantization, and trust_remote_code false.
Use `enforce_eager=False`. An eager diagnostic is allowed only as a separately
labelled failure investigation, never a production denominator.
Persist every prompt ID, output ID, resolved engine configuration, dtype,
plugin backend predicates, and exit status. Successful generation establishes
gateability only. Do not infer token parity from text or token counts.

Run the current pinned packed-recurrence test file in this environment.
Preserve its parameters, fixtures, tolerances, skipped cases, and exit status.
No skipped case is a pass. This file's tests include FP16/BF16/FP32, strided
inputs, padded state slots, grouped heads, output/state comparisons, and FP32
beta. These results do not substitute for the model token gate.

All child processes have finite timeouts. Stream logs to local files and
periodically preserve them to the supplied evidence directory. Stop on live
GPU-fault diagnostics and preserve partial captures. Do not reset the GPU,
clear quarantine, kill unrelated processes, or delete old installations.
Fail on exhausted memory/disk headroom instead of risking the worker.
Use existing managed-process helpers where their contracts fit this worker.

## Tests, gates, and acceptance

The fresh implementer first captures focused failure through the worker's
real CLI. CPU tests simulate external builds and model execution, never claim
GPU coverage. Test lease refusal, wrong hashes/revisions, unsafe archive paths,
stale build state, output reuse, wrong runtime/device, partial generation,
command failure, timeouts/fatal diagnostics, and evidence preservation.
Mutate each guarantee in scratch and require focused failure and restoration.

Focused gate: `python3 -m unittest tests.tools.test_strix_vllm_oracle`.
Run full `scripts/agent-preflight.sh` at the immutable head. A fresh reviewer
checks that head and repeats independent scratch mutations. The operator
reruns focused and applicable repository gates before executing the worker.
Report argument-dependent skips individually, not as a green full gate.

The operator owns a bounded resource-controller lease and runs build, model
generation, and upstream tests serially. Accept current-pin gateability only
after the pinned source builds and all six outputs contain 48 tokens with the
required identities. Report upstream tests separately. Preserve failure logs
without claiming the baseline is complete. Finish with an Outcome section
that records measurements, rejected choices, and remaining obligations.

## Risks and stop conditions

The new pin changes dependencies and model code. Model execution with the
plugin remains unmeasured. A dependency or compiler failure can block the run,
but does not prove the architecture unsupported. Resolve packaging failures in isolation.
Stop for missing authority, unavailable assets, lost lease, GPU fault,
unhealthy device, identity mismatch, or required changes outside this scope.
No compiler success establishes performance or numerical parity.

## Worker interface and dependency evidence

The implementation entry point is
`tools/bench/strix_vllm_oracle/worker.py`. Invoke it inside the operator's
lease with `--phase build --manifest PATH --output UNUSED_PATH`. Run with
`--phase run --manifest PATH --state BUILD_OUTPUT/build-state.json
--output OTHER_UNUSED_PATH`. Stage `runtime.py` beside the worker and retain
both bytes between phases; build state binds both hashes. State names the
unique local build directory, which must remain available for the run.

Manifest schema 1 contains these fields:

- `sources.vllm` and `sources.plugin`: `path`, `revision`, `sha256`.
- `model`: `path`, `bytes`, `sha256`, with the fixed artifact above.
- `assets`: file records with `path`, `bytes`, `sha256`, `relative_path`.
  Required relative paths are `model/config.json`, `tokenizer/config.json`,
  `tokenizer/tokenizer_config.json`, `tokenizer/tokenizer.json`, and
  `mmproj/mmproj-BF16.gguf`. Include every tokenizer sidecar in the manifest.
- `dependencies.requirements`: a hashed file record for bootstrap pins.
  `dependencies.indexes` lists explicit HTTPS indexes.
  `dependencies.expected_versions` binds `torch`, `torchvision`,
  `triton-rocm`, and the imported `triton_runtime` version.
  Optional `dependencies.wheels` contains hashed wheel records copied to
  an owned local find-links directory. Optional `source_archives` contains
  hashed file records with a unique dependency `name`, such as `amdsmi`.
  Dependency archives expose their build metadata at the archive root;
  a leading `./` is supported.
- Optional `limits` overrides positive `build_timeout`, `run_timeout`,
  `test_timeout`, `min_mem_bytes`, `min_disk_bytes`, and `max_log_bytes`.
  Defaults are 7200/3600/1800 seconds, 6 GiB available memory, 20 GiB free
  disk, and 512 MiB aggregate command logs. These are safety limits, not
  performance settings. The monitor terminates only its own process group.

The worker provisions the pinned source's actual pyproject build requirements.
It builds source wheels without build isolation, then resolves their runtime
dependencies under the explicit Torch constraints. Pip reports retain
resolved URLs and artifact hashes; freeze, pip check, wheel hashes, source
inventories, and extension inventories accompany the build state. Inherited
pip/Python installation destinations and pip configuration cannot redirect
installation into a donor environment. Old packages are not a compatibility
oracle: operator wheel metadata showed that torchvision 0.29 requires Torch
2.14, whereas torchvision 0.28 accepts the required Torch 2.13. The operator
manifest binds the compatible candidate versions. Overlap between plain
`triton` and `triton-rocm` requires the explicit byte-provenance envelope below;
without that envelope it remains a refusal, not an ignored pip-check error.
The captured build resolved the runtime dependencies and passed `pip check`
before and after compiler selection. Execution on the required model remains
unmeasured.

The packed-test adaptation copies the pinned file byte-for-byte to an isolated
test directory, avoiding unrelated repository-root conftest dependencies.
The file has three functions and eight parameterized cases, not 23 cases.
Success requires eight executed cases and zero errors, failures, or skips.
The runtime records read-only `LLM.apply_model` projection metadata using
`entrypoints/llm.py:599` at the pin. Parameter dtypes and shapes do not establish
the projection output dtype; that capture remains explicitly PENDING.

## Outcome

The harness establishes CPU-tested orchestration and a completed current-pin
GPU build. Build job `435bc0f3-638a-4812-87bc-8d44619e6d1b` exited 0.
Its evidence directory is
`/mnt/nas_share/rc/strix-vllm-3043.GkMABu/build-9624441-04/`.
`build-state.json` records `BUILT`. The worker and runtime hashes match
commit `9624441824915226c9412155ddffc6e3cf45d185`.

`build-identity.json` records ROCm on `gfx1151`, vLLM runtime
`0.28.1rc1.dev132+ge126687a9`, distribution
`0.28.1rc1.dev132+ge126687a9.rocm724`, and GGUF plugin `0.0.5`.
Both wheel builds and both `pip check` commands exited 0.
The compiler probes resolved
`/tmp/strix-vllm3043-7r3dipwh/venv/lib/python3.12/site-packages`.
Selected Triton `3.8.0` passed namespace verification. `triton-rocm=3.7.1`
remains nonauthoritative metadata. All three extension-target checks passed.
The command logs, exit records, source inventories, and wheel hashes remain
in the evidence directory.

The build state binds the normalized manifest SHA256
`c7c358b46c73cd2e170281e42cea11fafca54a8f5ea9c53fad6ce1a7d434ff0b`.
The retained `manifest.json` has raw-file SHA256
`499725007bd7b423265405e5dcd59439ed834a6a968feed06de14ea4b9a32472`.
These hashes cover different bytes. Only the normalized digest binds the state.

Runtime job `112a91ad-e51a-4627-aeae-61e2c0ed9643` uses this build state.
At 23:09 UTC on 7 September 2026, sibling output `run-9624441-01/`
contained runtime identity evidence but no `result.json`.
The six-prompt model run and eight packed numerical cases remain PENDING.
Issue #3043 remains open for these runtime acceptance requirements.
No throughput, token-parity, or model-dtype result is accepted by this change.
Production compilation remains enabled because it defines the denominator.
The token gate remains FAIL from the survey; successful counts cannot repair it.

The CLI test-first failure is retained in `/tmp/strix3043-red.log`; later red
cases cover inherited install destinations, termination, continuous output,
and partial capture in `/tmp/strix3043-red-{pip-target,signal,stream,partial}.log`.
Focused green evidence is `/tmp/strix3043-green-final.log`. Scratch mutation
commands and per-case logs are retained under `/tmp/strix3043-mutations*` and
`/tmp/strix3043-mutation-*.log`, with byte-for-byte restoration assertions.
The immutable-head full preflight is reported separately in the implementer
handoff, including every argument-dependent SKIP. CPU fixtures substitute
only external commands and tiny pinned artifacts; they execute the real
worker CLI and a separately tested runtime CLI.

## Review repair: guard coverage and finite limits

Fresh review of `799901422` found ten guards that the complete CPU suite
did not detect when removed. Add CLI refusal cases for schema, worker/runtime
state hashes, changed extracted sources, fixed model identity, corruption
after model copying, missing required assets, imports outside the virtual
environment, local-state boundaries, copied upstream-test integrity, and
missing resolved dtype. Assert the specific refusal and absence of later
commands. Copy-corruption fixtures alter the destination after a real copy;
they do not substitute a pre-copy checksum failure.

Five further runtime mutations escaped the original suite. Extend its real
runtime CLI fixtures to reject imports outside the environment, non-ROCm
platforms, a non-gfx1151 device, and missing GGUF plugin registration. Supply
a nonempty model projection fixture and assert captured metadata so deleting
`LLM.apply_model` cannot pass. Keep all runtime production code unchanged.

The review also found that nonfinite numeric limits pass the positive-number
check. JSON accepts NaN and Infinity, and `1e309` parses as infinity. These
values disable comparisons that enforce deadlines and resource bounds.
Capture real CLI failures for these values before adding finite-number
validation. Preserve positive integer and finite fractional limits. Do not
change defaults, dependency policy, or runtime behavior beyond this guard.

Record original-suite mutation survivors, focused green, and scratch failures
for every repaired guarantee. Restore scratch bytes after each mutation.
Run the full preflight on the final immutable head after review findings are
complete. The operator still owns dependency decisions and hardware execution.

## Selected Triton compiler provenance

The operator selected this bounded packaging mechanism for #3043. The manifest
adds `dependencies.selected_triton`: a hashed wheel record with
`distribution=triton` and `version=3.8.0`. Its measured wheel is
`triton-3.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`,
247,972,921 bytes, SHA256
`e91ffa46d095b252248297292dd22bcbacd53a125a0c2eefbbbf74925a320bc3`.
The operator verified all 419 signed RECORD entries and 414 namespace files,
with only `triton/` and `triton-3.8.0.dist-info/` roots and no symlinks.
`expected_versions.triton=3.8.0` and `triton_runtime=3.8.0` bind the selected
compiler. `triton-rocm=3.7.1` remains installed resolver metadata only and is
nonauthoritative for imported compiler bytes.

The initially selected ROCm 3.7.1 wheel (311,741,985 bytes, SHA256
`b200ade2418b8450e4e3e69c33c804e2cc07d1e36dc9678ac8ac1dd65f9e10e1`)
fails strict RECORD validation. It contains unrecorded AMD libraries
`libelf.so`, `libnuma.so`, and `libtinfo.so`, while RECORD names absent
`libelf.so.1`, `libnuma.so.1`, and `libtinfo.so.6`. Actual versus recorded
sizes are 114625/109000, 57297/51400, and 194537/187552 bytes, respectively;
hashes differ too. Reject that wheel rather than renaming files or weakening
RECORD validation. The operator selected the valid plain 3.8.0 wheel instead;
this changes the hash-bound input and metadata roles, not the provenance
mechanism or vLLM source pin.

After runtime dependency installation and successful `pip check`, validate
the selected wheel hash, metadata, RECORD hashes and sizes, and every archive
path. Reject escaping paths, symlinks, missing records, and unexpected roots.
Before importing any runtime package, move the resolver-created `triton`
namespace intact into unique quarantine under the newly owned local build.
Retain its provenance and populate a fresh namespace only from the selected
wheel. Do not alter distribution metadata or any donor environment. Run
`pip check` again after this selection.

Verify every installed compiler file against the wheel, including native
backend files. Reject missing, changed, additional, or symlinked files,
including preexisting Python bytecode caches. Set `PYTHONDONTWRITEBYTECODE=1`
for worker children so selected compiler imports do not add unverifiable
executable cache bytes. Record the selected
wheel identity, full namespace inventory, imported module path and version,
AMD backend/native identities, and both distribution metadata versions.
Explicitly label nonselected metadata as nonauthoritative for imported bytes.
Recheck compiler state before run imports and after generation and tests.
Overlap without this complete proof remains a failure.

Preserve safe regular-file permission bits from the selected wheel, including
its executable tools, without propagating setuid or setgid bits. Bind and
recheck those modes alongside file hashes. A tiny executable wheel fixture
must run after selection and fail its real CLI gate when mode preservation
is removed. The selected wheel includes executable NVIDIA tools; preserving
their installation fidelity does not claim new backend support.

Use tiny wheels with valid RECORD metadata in CPU CLI tests. Capture a red
successful-selection fixture with overlapping metadata before implementation.
Cover wrong wheel, malformed RECORD, namespace changes, extra files, symlinks,
wrong import paths, and changed state. Mutate every guarantee in scratch and
restore its bytes. This envelope does not advance source pins, waive tests,
or establish GPU gateability. The worker and runtime changes are limited to
this envelope and the finite-limit repair; dependency choices remain explicit.

## Launch and review repair evidence

The controller's measured visibility value triggered the blanket `CUDA_`
refusal before manifest loading. The test-first failure is
`/tmp/strix3043-repair2-visibility-red.log`. The worker now accepts exactly
`CUDA_VISIBLE_DEVICES=0` after checking the named lease and job identity.
Both phases retain this value in their environment evidence and child processes.
Other visibility values and inherited experimental settings remain refusals.

Fresh review identified nine compiler provenance guards without mutation
coverage. Valid tiny-wheel fixtures now isolate metadata presence and identity,
duplicate archive members, malformed or duplicate RECORD rows, and required
compiler files. Same-byte, same-mode links isolate both namespace link guards.
The run tests refuse a changed quarantine and quarantine outside owned storage.
The build test refuses an installation site outside the new environment and
checks that the donor remains intact. The runtime test binds all six prompt-ID
lists independently to historical `gen_rocm.py` at #2740.

Original-suite survivor reproductions are retained in
`/tmp/strix3043-repair2-baseline-*.log`. Scratch mutation programs are
`/tmp/strix3043-repair2-mutations.py` and
`/tmp/strix3043-repair2-launch-mutations.py`. Each program asserts byte-for-byte
restoration after each mutation. Guard removals, broader visibility acceptance,
lost child visibility, omitted environment evidence, and changed historical IDs
fail their focused assertions. The focused suite has 42 tests. Final focused
evidence is `/tmp/strix3043-repair2-focused-final-v2.log`. The immutable-head
preflight result and argument-dependent skips remain in the implementer handoff.

No runtime production code, compiler packaging contract, source pin, or model
workload changes in this repair. GPU gateability remains PENDING.

## Explicit virtual-environment scheme repair

The developer approved autonomous repair after the namespace diagnosis.
Lease `4ade5e61-c10a-4e98-8b1b-3da36018dc97` measured Python 3.12 on Strix.
Normal isolated startup resolves `purelib` through the `venv` scheme.
Startup with `-I -S` instead selects Ubuntu's `posix_local` scheme.
Passing `base` and `platbase` does not change that selected scheme.
The worker therefore checks `venv/local/lib/python3.12/dist-packages/triton`.
The installed directory is `venv/lib/python3.12/site-packages/triton`.
An explicit `scheme="venv"` produces the installed path with `-I -S` intact.
Evidence: `/tmp/strix3043-namespace-diagnosis.log`.

Repair only `Session.compiler_namespace` and its CPU regression coverage.
Select the `venv` scheme explicitly and retain the supplied environment bases.
Keep `-I -S`, path ownership checks, and all compiler provenance checks.
Do not import unverified runtime modules to discover their installation path.
Do not add fallback directories, bypass symlink checks, or relax wheel checks.

The failing regression must execute the actual emitted Python expression.
Model a distribution whose default scheme differs from its virtual-environment
scheme without computing the expected path from the worker under test.
Exercise the production worker entry point and preserve the existing CLI cases.
Capture the wrong-path failure before implementation, then focused green.
Independently remove the explicit scheme and require the regression to fail.
Mutate loss of `-I`, `-S`, or either supplied base and test the intended guard.
Restore scratch bytes after every mutation. Run the full preflight on the
immutable repair head, obtain fresh review, and let the operator repeat gates.

Build attempt `c53f6048-ae96-4097-8ed6-29e8d9118e7f` built vLLM and its
GGUF plugin, installed dependencies, and passed `pip check` before this failure.
Its artifacts remain evidence, not a successful build-state certificate.
This repair adds no resume mode and never fabricates that certificate.
The operator retains the wheels and can run the repaired worker in a fresh
environment with verified read-only dependency inputs. Source identity,
extension targets, compiler provenance, model outputs, and upstream tests
remain mandatory. Model gateability and performance remain PENDING.

### Namespace repair evidence

`Session.compiler_namespace` now supplies `scheme='venv'` explicitly.
The CPU regression executes its emitted expression in a real interpreter
through both worker CLI phases. The fixture gives the default and `venv`
schemes separate dictionaries, including on interpreters where they alias.
Only the distribution scheme table changes. The real `sysconfig.get_path`
expands the path, with assertions on isolation flags and supplied bases.

The test-first command was `python3 -m unittest
tests.tools.test_strix_vllm_oracle.OracleCliTests.test_cli_executes_compiler_site_expression_with_distinct_default_scheme`.
It exited 1 with `unsafe selected Triton namespace` and the default scheme's
`venv/local/lib/dist-packages` path. Evidence: `/tmp/strix3043-namespace-red.log`.
The same test passed after the explicit scheme selected `venv/lib/site-packages`.
The fixture keeps its existing abbreviated package layout, independent of
the worker expression and the host's Python version.

Scratch mutations removed the scheme, `-I`, `-S`, `base`, and `platbase`
independently. Each exited 1. The scheme mutation reproduced the wrong path.
The other four failed their named isolation or base assertion.
Logs: `/tmp/strix3043-namespace-mutation-{scheme,I,S,base,platbase}.log`.
`cmp` verified byte restoration after each mutation and exited 0.
The complete focused gate and immutable-head preflight are reported in the
implementer handoff. GPU execution remains the operator's obligation.

## Owed

Issue #3048 owns the named metadata RPC repair specified below. It must land
with its regression and independent review before retrying generation.

Issue #3043 still owes production-mode generation for all six specified prompts,
with 48 tokens each. Eight upstream packed-test cases must execute without
errors, failures, or skips. The operator must retain the resolved configuration,
compiler provenance, output tokens, projection metadata, and exit statuses.
Successful generation counts establish no token parity. The token gate remains
FAIL under #2534.

The packed ROCm port follows under its own issue and committed spec after
this prerequisite. Its tests must follow current vLLM rather than old CUDA
goldens. GGUF optimization remains #3016/#3017/#3018. Valid matched profiler
timestamps remain #3040. Model correctness remains #2534 and the owning arm.

## Named metadata RPC repair (#3048)

### Measured failure and upstream anchors

Run `112a91ad-e51a-4627-aeae-61e2c0ed9643` reached model loading and production
graph compilation. Generation exited 1 after 412.604 seconds before any output
tokens. `LLM.apply_model(projection_metadata)` sent a Python function through
the default serializer, which refused it. Evidence remains in
`run-9624441-01/logs/generation.log` and `run-9624441-01/failure.json`, beside
the completed build evidence. The old fake `LLM.apply_model` accepted the
function directly and therefore did not reproduce this failure.

All upstream anchors use `e126687a9a828d513c01a07cd69f025f27d63280`:

- `vllm/entrypoints/llm.py:567`: `collective_rpc` accepts a method name.
- `vllm/v1/worker/worker_base.py:145`: `get_model` supplies the worker model.
- `vllm/v1/worker/worker_base.py:285`: `worker_extension_cls` resolves a named
  class and adds its nonconflicting methods to the worker.
- `vllm/v1/serial_utils.py:221`: the default encoder rejects arbitrary objects.
- `tests/v1/test_serial_utils.py:271`: the no-pickle test expects `TypeError`.

`git log -S VLLM_ALLOW_INSECURE_SERIALIZATION` identifies upstream
`6930a41116`, vLLM #17490, as the introduction of the explicit opt-in.
Do not enable that opt-in. The user directed this repair on 7 September 2026.

### Design, constraints, and stop conditions

Keep the projection inspection helper and its returned metadata. Add one
worker-extension class in the already hash-bound `runtime.py` module. Its
uniquely named method calls the helper on `self.get_model()` inside the worker.
Configure that class by its importable qualified name, not a class object.
Call `LLM.collective_rpc` with the method name and no callable arguments.
Only strings, lists, dictionaries, and primitive metadata cross the RPC.
The named extension is diagnostic instrumentation, not a model replacement.
Record its class in the resolved engine arguments without changing any
production compilation, sampling, model, or quantization default.

Keep projection output dtype explicitly PENDING. Do not replace parameter
metadata with model configuration, remove the inspection, swallow its errors,
or add a fallback to insecure serialization. Do not alter worker build-state
identity checks, compiler checks, the oracle pin, or the eight upstream cases.
A changed runtime hash requires a new certified build. Do not edit a completed
build state to make old evidence accept new code.

Stop for a required insecure setting, unimportable extension, missing model
interface, or a change outside the runtime adapter and its CPU regression.
The saved GPU failure is the real-oracle red result. CPU green does not prove
that the repaired adapter executes on the real worker.

### Task 1: use the named worker RPC {id: 1, deps: []}

Files: `tools/bench/strix_vllm_oracle/runtime.py`,
`tests/tools/test_strix_vllm_oracle.py`, and this spec's repair evidence only.
The spec is committed before implementation. Use a fresh implementer and
fresh independent reviewer, followed by the operator's own gate.

First reproduce the failure through the real runtime entry point. The fake
serializer must refuse a callable under the pinned default, as the upstream
no-pickle test requires. Preserve that upstream revision and document the
CPU-only adaptation rather than claiming a real GPU serializer test.
Resolve the configured extension by its actual qualified name and execute its
method on a worker fixture. Assert the independent expected parameter dtype,
shape, quantization method, output-dtype limitation, and all six output lists.
Assert no insecure setting is introduced and RPC failures remain failures.

Capture red before implementation, then focused green. In scratch, restore
the callable path, remove extension configuration, remove the named RPC call,
and remove or corrupt the returned parameter metadata. Each mutation must fail
its intended assertion. Restore the scratch bytes after every mutation.

Verify: `python3 -m unittest tests.tools.test_strix_vllm_oracle`.
Run `scripts/agent-preflight.sh` on an unchanged final head. Use a disk-backed
temporary filesystem with adequate space, because the full tools suite
explicitly tests file-cache eviction and tmpfs does not satisfy that contract.
Report argument-dependent skips separately. Preserve test and review evidence
here; retain #3043 for real model execution and matched benchmark acceptance.
