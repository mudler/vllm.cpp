# Repair the ROCm skinny sacred-gate provenance

## Now

Issue [#2772](https://github.com/mudler/vllm.cpp/issues/2772) belongs to Row
`BACKEND-ROCM`. The issue is open, and its default checkpoint gate is
**FAILING** at the reviewed base `3ab4b1209be19faaf734697532d4d1eb1722066c`.

The first failure is prompt 10, generated token 10. Default local wvSplitK
emits token 369, but the committed local anchor records token 488. Setting
`VT_ROCM_SKINNY=0` restores the old anchor and 137/137 assertions. A locked
bisect identifies `f38c1edc4ea679348f856c0c0b20fb0702f77daf` as the first bad
commit. That commit made the existing wvSplitK arm the eligible default.

The pinned production oracle now runs. It chooses token 369 at the disputed
prefix and supports every token in the current local 16 by 16 sequence with a
zero teacher-forced gap. The selected repair is therefore minimal
rederivation. Preserve the default wvSplitK path, make both capture scripts use
production vLLM by default, and refresh the four ROCm arrays and their
provenance.

This specification and its implementation use one pull request. This row is
not `DONE`, so this specification has no `## Outcome` section.

## Goal

Make the Qwen3.5-0.8B ROCm sacred gate reproduce the pinned vLLM production
mode that governs acceptance. Keep the existing checkpoint path and acceptance
policy causal to default local wvSplitK.

The completed implementation must have these properties:

1. Both oracle scripts default to production mode and expose eager mode only
   through explicit `--enforce-eager`.
2. A CPU test distinguishes default production mode from explicit eager mode
   for both scripts.
3. Exactly four ROCm arrays derive from the retained production evidence.
4. The existing real `LoadedEngine::FromModelDir` case passes 137/137 and
   reports 15/16 strict prompts, one oracle-tied prompt, maximum gap 0.0, and
   zero forward divergence.
5. Routing `WvSplitKBT` to the existing BLAS fallback makes that real case fail
   at prompt 10, token 10.
6. The existing prerequisite exit contract and wvSplitK focused test remain
   unchanged.

## Selected decision

The evidence selects rederivation. It does not select a kernel repair, an
eligibility change, a default change, or a performance campaign.

At the first shared-prefix disagreement, pinned vLLM reports equal log
probabilities for tokens 369 and 488:

```text
369  -1.2000471353530884
488  -1.2000471353530884
```

Production greedy and teacher-forced argmax both choose 369. The current local
prompt-10 sequence matches the production oracle:

```text
279 2691 13 271 248068 271 248069 271 760 4952 369 2972 1802 159034 271 760
```

The eager diagnostic reproduces the historical committed oracle output. Thus
the stale evidence came from capture mode, not restored dependency drift.

## Verified evidence

### Oracle identity

`VLLM_ORACLE` resolves to
`/home/vikash/oracle/vllm-rocm-oracle-python`. The wrapper uses network-disabled
image ID
`sha256:80aab4c182a1f3eeebe286173977e57fcaf10a049b41f475655b35d285de31dc`.

The oracle identity is:

| Field | Value |
|---|---|
| vLLM runtime | `0.23.1rc1.dev1511+g555967922` |
| vLLM source | `5559679229bc961848b121ccdeaa8fa5d79bec98` |
| Wheel SHA-256 | `c4892c729daaff1ac3e8ad72d4e4753118b84e7d7c265634a1326b261a44ee43` |
| Base image digest | `sha256:ba82bf93fc2d3787550b77fb2a79fe230130f05bec737c671093251c55459309` |
| HIP userland | `7.2.53211` |
| Device | `gfx1100` |
| GPU exclusion | `/home/vikash/gpu.lock` |

The model is
`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`.
Both engines use these exact local files:

| File | Bytes | SHA-256 |
|---|---:|---|
| `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| `model.safetensors.index.json` | 50,900 | `d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e` |
| `config.json` | 2,907 | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |

### Production results

The production oracle ran with `enforce_eager=False`. Its compilation mode was
`VLLM_COMPILE`, and its graph mode was `FULL_AND_PIECEWISE`.
`VLLM_ROCM_USE_SKINNY_GEMM` was unset and resolved true.

The complete teacher-forced run compared all 256 current local prefixes. It
found zero prompt mismatches, zero local tokens outside the oracle top 20, 256
strict argmax matches, zero nonzero gaps, and maximum gap 0.0. Evidence is
`/tmp/qwen35-skinny-oracle-full-gap-2772-production/result.json`, SHA-256
`495ff7624c7a46fd135edc18fddc364ef449a10db78a5eb3e5b7d2fc1a99589b`.

The production K=10 greedy run was deterministic in all 256 cells. It differs
from current local output only at prompt 7, positions 8 through 15, after an
exact tie at the first split. Evidence is
`/tmp/qwen35-skinny-oracle-greedy-k10-2772-production/result.json`, SHA-256
`92460eb055228491c7e30c8bef83d7ce4a909db1c6c925ef31d0fba4fdf4c5d4`.

The four retained candidates are:

| Destination array | Retained source | SHA-256 |
|---|---|---|
| `our_ids.npy` | `qwen35-skinny-oracle-full-gap-2772-production/our_ids.production.npy` | `395ecea93be0c2cf036a0d8d515493a5178a5dcaea172df8ff71cfd80ab24112` |
| `neartie_gap_mnats.npy` | `qwen35-skinny-oracle-full-gap-2772-production/neartie_gap_mnats.production.npy` | `c707d168d23aea394987c4a40c4e92d8347c0400e9790df0851c94d37c113a82` |
| `greedy_ids.npy` | `qwen35-skinny-oracle-greedy-k10-2772-production/greedy_ids.production.npy` | `c120eec482e2029037269f55721f9100c0f4717c956017aeaf51346abfd4b9c8` |
| `greedy_dist.npy` | `qwen35-skinny-oracle-greedy-k10-2772-production/greedy_dist.production.npy` | `1924f7a8915d367ea462975d0b5f91da78ba0f36d922c0d563abff9457ec16c8` |

The expected differences from the committed arrays are exact:

- `our_ids.npy` changes six cells at prompt 10, positions 10 through 15.
- `neartie_gap_mnats.npy` changes `(7,8)` and `(10,10)` from 125 to 0.
- `greedy_ids.npy` changes 16 cells at prompts 7 and 10, positions 8 through
  15.
- `greedy_dist.npy` changes 160 cells.

These hashes belong to this evidence section and issue comments. Do not add an
array-hash inventory or copy these hashes into another tracked file.

### Eager diagnostic

The committed `scripts/qwen3-oracle-capture.py` hard-codes
`enforce_eager=True`. Running it unchanged in the restored image reproduced the
historical `greedy_ids.npy` byte for byte, with SHA-256
`865293612ec0e5da77a188c8513cca80f0efcb4235fae37b380530a0ca2413b3`.
The runtime reported disabled compilation and graph mode `NONE`.

This result isolates execution mode as the provenance defect. Eager remains a
useful diagnostic, but it is not the production denominator.

### Actual engine configurations

The engines did not use an identical capacity configuration. Record their
actual defaults separately.

| Setting | Local sacred case | Production vLLM oracle |
|---|---|---|
| Construction | `EngineParams{}` | production `LLM` |
| Model dtype | BF16 | BF16 |
| Block size | 32 | runtime default, not an identity condition |
| KV blocks or tokens | auto, fallback 256 blocks | profiled 1,085,870 GPU KV tokens at explicit `gpu_memory_utilization=0.80` |
| Maximum sequences | 32 configured, 15 resolved | 256 |
| Maximum batched tokens | 2,048 for the dense registration | 8,192 |
| Maximum model length | 8,192 resolved | 262,144 |
| Prefix caching | hybrid-model default off | off |
| KV-cache dtype | not characterized in this issue | reported as `auto` |
| Oracle execution mode | n/a | `enforce_eager=False`, `FULL_AND_PIECEWISE` |
| Skinny mode | `VT_ROCM_SKINNY` unset, default on | `VLLM_ROCM_USE_SKINNY_GEMM` unset, resolves true |

The comparison requires identity only for the model bytes, exact numeric
prefix, one request, one next token, greedy sampling, and default skinny mode.
Do not require shared scheduler capacity values. Do not claim a physical cache
dtype from `auto`. Issue #2773 owns cache and state characterization.

### Executing chains

The pinned upstream source chain remains:

```text
vllm/envs.py::VLLM_ROCM_USE_SKINNY_GEMM
  -> vllm/model_executor/layers/utils.py::rocm_unquantized_gemm_impl
  -> vllm/_custom_ops.py::wvSplitK
  -> csrc/rocm/torch_bindings.cpp::wvSplitK
  -> csrc/rocm/skinny_gemms.cu::wvSplitK
  -> csrc/rocm/skinny_gemms.cu::wvSplitK_hf_sml_
```

The local production chain remains:

```text
src/vllm/entrypoints/model_loader.cpp::LoadedEngine::FromModelDir
  -> src/vllm/model_executor/models/model_registry.cpp::ModelRegistry::Forward
  -> src/vllm/model_executor/models/qwen3_5.cpp::MatmulBTRawD
  -> src/vt/ops.cpp::MatmulBT
  -> src/vt/rocm/rocm_matmul_hipblaslt.hip::MatmulBTKernelRocm
  -> src/vt/rocm/rocm_matmul_hipblaslt.hip::WvSplitKBT
  -> src/vt/rocm/rocm_skinny_gemm.hip::wvSplitKSml
```

The source audit and retained production run establish the selected default.
No direct custom-operation replay or raw-BF16 byte-identity condition remains.

## Scope

### In scope

- Change `scripts/qwen3-oracle-capture.py` and
  `scripts/qwen3-neartie-gap.py` to default to production mode.
- Add explicit eager diagnostics to both scripts.
- Add one CPU test for both scripts' mode selection and narration.
- Refresh exactly four arrays under
  `tests/parity/goldens/qwen35_greedy_0_8b/`.
- Update that directory's `manifest.json` and the provenance comments in
  `tests/parity/test_qwen35_paged_engine.cpp`.
- Preserve the real checkpoint case, its acceptance logic, and its prerequisite
  behavior.
- Run the existing relevant wvSplitK test without changing it.

### Out of scope

- Any kernel, default, eligibility, arithmetic, or performance change.
- A shipped observer, execution-shape freeze, operand dump, logit dump, or
  direct custom-operation replay.
- An eighth upstream shape or a broader upstream test port.
- Cache dtype, capacity, state-space, convolution, attention, or layer-numeric
  characterization.
- Product code, a framework seam, a public API, or public documentation.
- Tenstorrent arrays, prompt-ID files, unrelated goldens, and threshold changes.

If a focused result contradicts the zero-gap production evidence, stop with
`NEEDS_DECISION`. Do not reactivate the removed diagnostic campaign silently.

## Implementation contract

### Make production mode reproducible

Both scripts must omit `enforce_eager` or pass `False` by default. Both scripts
must accept an explicit `--enforce-eager` flag that passes `True` to `LLM`.
Each script must print the selected mode in words before engine construction.
The narration must distinguish production from eager and state the resolved
`enforce_eager` value.

A small pure argument or keyword-construction helper is allowed. It remains
private to the scripts and does not create a framework or public API.

### Add the smallest CPU mode test

The future implementation adds
`tests/scripts/test_qwen3_oracle_modes.py`. This is an implementer-added test;
it does not exist in this specification commit.

The test imports each script without constructing vLLM. For each script, it
must prove:

- no flag selects production and yields absent or false `enforce_eager`;
- `--enforce-eager` yields true `enforce_eager`; and
- the selected narration names production or eager correctly.

Run this test against the current hard-coded eager scripts before changing
them. It must fail for the intended default-mode reason. It must pass after
both scripts change.

### Refresh only the selected evidence

Replace exactly these files in
`tests/parity/goldens/qwen35_greedy_0_8b/`:

```text
greedy_ids.npy
greedy_dist.npy
our_ids.npy
neartie_gap_mnats.npy
```

Use the retained sources and expected hashes in `## Verified evidence`. Verify
the exact old-to-new cell counts before staging.

Update `tests/parity/goldens/qwen35_greedy_0_8b/manifest.json` with the restored
image, source, wheel, base image, HIP, date, production mode, graph mode,
default skinny mode, commands, and expected gate summary. Do not add array
hashes to the manifest.

Update the provenance comments in
`tests/parity/test_qwen35_paged_engine.cpp`. The comments must identify
production capture as the denominator and eager as diagnostic history. Do not
change executable logic, thresholds, cases, or prerequisites.

### Preserve the causal sacred case

Keep the single existing doctest case in
`tests/parity/test_qwen35_paged_engine.cpp::RunGate`. It must continue to enter
through `LoadedEngine::FromModelDir`, run the 16 prompts, check the local anchor
before the near-tie policy, and prove native provider selection.

Do not add an observer or freeze internal GEMM shapes. The refreshed anchor is
causal without either mechanism. In a scratch copy, route the eligible
`WvSplitKBT` call in
`src/vt/rocm/rocm_matmul_hipblaslt.hip::MatmulBTKernelRocm` to its existing
BLAS fallback. The real gate must fail at prompt 10, token 10, because fallback
emits 488 instead of refreshed anchor token 369. Restore the file byte for
byte, then require the gate to pass again.

### Preserve prerequisite behavior

Do not edit
`tests/parity/test_qwen35_paged_engine_prerequisites.cmake` or its CTest
registration. The prerequisite probe contract remains:

| Probe | Required exit |
|---|---:|
| missing `greedy_ids.npy` | 77 |
| missing `our_ids.npy` | 77 |
| missing `neartie_gap_mnats.npy` | 77 |
| all three present | 86 |

Exit 86 is the complete-set sentinel. It is not a successful model run. A
zero-case `RC 0` is invalid for the dedicated real gate.

### Preserve the existing kernel test

Run the unchanged case
`tests/vt/test_backend_cross_device.cpp::decode-skinny MatmulBT (wvSplitK path) matches the CPU oracle`.
Do not add `(2,10240,1024)` or another eighth upstream shape. Issue #487 owns
broader routing, kernel coverage, and performance work.

## Acceptance ledger

| Obligation | Current result | Completion result |
|---|---|---|
| Default sacred gate | **FAILING** at prompt 10, token 10 | 137/137 with the exact summary in `## Goal` |
| Production oracle identity | **SATISFIED** | Preserve the recorded pins and mode |
| Selected decision | **SATISFIED** | Minimal rederivation only |
| Script reproduction mode | **FAILING** because both scripts hard-code eager | Default production plus explicit eager for both scripts |
| Four ROCm arrays | **FAILING** because they are eager-era evidence | Exact retained production candidates |
| Prerequisite contract | **SATISFIED** | Preserve 77/77/77/86 |
| Existing wvSplitK test | **SATISFIED** at the reviewed source | Pass unchanged; no eighth shape |
| Causal production route | **PENDING** reviewer mutation | BLAS routing makes the sacred case red |
| Repository preflight | **FAILING** on inherited suites | `RC 0`, or report a set-identical inherited failure without calling it green |
| Upstream symbol anchors | **FAILING** on inherited anchors | `RC 0`, or report a set-identical inherited failure without calling it green |

The controlled pre-edit preflight at `3ab4b1209` returned `RC 1`. It reported
five failures: `test_agent_onboard`, `test_check_windows_portability`, trailer
suites, commit style suites, and tools suites. It also reported five
argument-dependent skips. These results are not green and are outside this
issue's authority.

The reviewed upstream symbol-anchor command returned `RC 1` with 15 stale or
unresolvable citations outside this specification. The specification's own
cited paths and symbols resolved. Re-run both gates and classify any remaining
failure. Do not absorb either inherited repair into #2772.

## Red and green evidence

### Red

- Run the implementer-added CPU mode test before editing either script. The
  current eager hard-coding must make the production-default assertion fail.
- Retain the existing default checkpoint failure at prompt 10, token 10, as the
  checkpoint red. Do not rerun it without the required GPU exclusion.

No manufactured kernel red is required because no kernel behavior changes.

### Focused green

Run these commands from the task worktree. The first test name is
implementer-added:

```sh
python3 tests/scripts/test_qwen3_oracle_modes.py

cmake --build build-hip --target test_qwen35_paged_engine \
  test_backend_cross_device -j 4

ctest --test-dir build-hip --output-on-failure \
  -R '^test_qwen35_paged_engine_prerequisites$'

env -u VT_ROCM_SKINNY \
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
  flock /home/vikash/gpu.lock \
  build-hip/tests/test_backend_cross_device \
  '--test-case=decode-skinny MatmulBT (wvSplitK path) matches the CPU oracle'

env -u VT_ROCM_SKINNY \
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
  flock /home/vikash/gpu.lock \
  build-hip/tests/test_qwen35_paged_engine
```

The prerequisite CTest must report all four expected child exits. The
unfiltered dedicated binary must run one real case, return `RC 0`, pass 137/137
assertions, and print this semantic summary:

```text
16/16 prompts PASS
STRICT token-exact vs oracle per-prompt greedy: 15/16
near-tie-band only: 1/16
max gap 0.0 nats
0 forward-divergent
```

The exact punctuation can remain the existing output. Zero cases with `RC 0`
does not satisfy this gate.

### Full verification

Run the controlled record and citation gates after focused green:

```sh
scripts/agent-preflight.sh
python3 scripts/check-symbol-anchors.py \
  --upstream-root /home/vikash/oracle/vllm-src
```

Record each return code and output path. `RC 0` is green. A nonzero result that
is set-identical to the controlled base is an inherited failure, not green and
not an implementation regression. Any new failure blocks completion.

The operator reruns the row gate. An implementer report does not replace that
run.

## Fresh review mutations

A fresh reviewer uses a scratch copy of the immutable implementation head. The
reviewer applies one mutation at a time:

1. Flip either script's default from production to eager. The CPU mode test
   must fail.
2. Route `WvSplitKBT` to the existing BLAS fallback. The real sacred case must
   fail at prompt 10, token 10.
3. Change one oracle-tied gap from 0 to 501 milli-nats. The real sacred case
   must report forward divergence and fail.
4. Exercise each missing prerequisite. Each child must exit 77.
5. Exercise the complete prerequisite set. The child must exit 86.

Do not mutate arithmetic, eligibility guards, thresholds, or unrelated routing.
After each mutation, restore every touched file byte for byte and verify its
SHA-256 before the next mutation. Re-run focused green on the restored head.

## Risks and controls

- An ordinary rerun can restore stale eager artifacts. The default-mode CPU
  test and script narration prevent silent mode reversal.
- A copied array can hide a wrong source. Exact retained hashes, cell counts,
  manifest provenance, and the real gate bind the refresh to production mode.
- A prerequisite regression can create a false pass. The registered CTest pins
  77/77/77/86 before the unfiltered real binary runs.
- A token gate can pass after bypassing a kernel. The BLAS call-site mutation
  proves the refreshed anchor needs default wvSplitK.
- Capacity differences can reject a valid oracle run. The comparison binds
  semantic workload inputs and records each engine's capacity separately.
- Inherited repository failures can look green when summarized loosely. Record
  their nonzero return codes and compare their exact failure sets.

## Stop conditions

Return `NEEDS_CONTEXT` if an artifact, model, oracle, source, or retained hash
does not match this specification.

Return `NEEDS_DECISION` if focused evidence contradicts the zero-gap production
result or requires any out-of-scope kernel, default, cache, or performance
change.

Return incomplete if the mode test, prerequisite CTest, unchanged wvSplitK
test, real sacred case, or required review mutation does not produce its exact
result. Do not weaken the anchor or near-tie acceptance logic.

Classify inherited preflight and symbol-anchor failures. They are not green,
and their repair is not authorized in this issue.

## Owed

- Issue [#2773](https://github.com/mudler/vllm.cpp/issues/2773), Row
  `BACKEND-ROCM`, owns cache, state-space, convolution, attention, and
  layer-numeric characterization. It remains blocked only until #2772 lands.
- Issue [#487](https://github.com/mudler/vllm.cpp/issues/487), Row
  `BACKEND-ROCM`, owns broader skinny-GEMM kernel coverage, routing, and
  performance work.

Neither issue is absorbed into #2772.

## Git integration

Use one pull request for this specification and its implementation. The pull
request body links issue #2772, carries `Row: BACKEND-ROCM`, and closes #2772
only after focused green, fresh review, and operator verification.

Do not edit public documentation or an application programming interface. Add
`## Outcome` only when the row reaches `DONE`.
