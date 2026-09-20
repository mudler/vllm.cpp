# Profile Strix Qwen3-4B workers from process start

Row: `BACKEND-GATE-ROCM-SGLANG`

Issue: [#3076](https://github.com/mudler/vllm.cpp/issues/3076).

## Now

The user approved worker-start profiling of both engines on 2026-09-19. This
document defines the diagnostic harness and evidence contract before its
implementation. It performs no GPU work, attributes no performance difference,
and changes no inference default, correctness threshold, or tolerance. The row
remains `INVENTORIED` until an implementation and its required review land.

Repository policy supplies the one-pull-request default for the later
implementation. No row-specific preference selects a split pull request. A
fresh implementer must work from this committed design, a fresh reviewer must
mutate the immutable implementation, and the operator must run every hardware
gate itself.

The worker-image campaign baseline is `PENDING`. Bootstrap can produce a
candidate, but it cannot accept one. The replay and hardware gates remain
`PENDING` until an operator commits the campaign commitment defined in this
document and a fresh reviewer passes that immutable head.

## Record placement

The campaign's earlier
`.agents/specs/strix-qwen3-4b-c4-performance.md` exists at commit
`b923ac2c4c24e9c608d4a6e02e868538be7536fa`, but not on current `main`.
That 555-line record includes an attach-first sequence and completed historical
campaign details. Restoring it would make stale campaign state current and
would obscure the narrower decision now being reviewed.

This file is the per-issue design for `ISSUE-GH-3076`. The generic owning row
continues to use `.agents/specs/competitive-benchmarks.md`; this design does not
change that row's lifecycle. This shape avoids an unrelated edit to a shared
keyed matrix and preserves the old record at its immutable commit. Inspect the
old record with:

```sh
git show b923ac2c4c24e9c608d4a6e02e868538be7536fa:.agents/specs/strix-qwen3-4b-c4-performance.md
```

## Problem

Two historical qualification corpora reported diagnostic concurrency-four
medians of 61.5947184118 output tokens/s for vllm.cpp and 79.5022690204 for
production vLLM. The token gate failed. These values are not accepted benchmark
results and cannot justify a product change.

The missing evidence is a complete matched trace from the actual GPU worker in
each production process tree. The earlier attach route passed only a safety and
identity preflight. It did not prove that late attachment observes HIP graphs
captured before attachment. Retained rocprofiler SDK source shows that a
previous `rocprofiler_configure` provider can activate profiling before the
attachment proxy queues are created. Retained process evidence identifies
Torch/Kineto as that provider in the production-vLLM target.

The earlier vLLM process-start experiment reached production
`FULL_AND_PIECEWISE` graph capture for sizes 1, 2, 4, and 8, but artifact
finalization failed. Its worker result grew to 490,690,366 bytes, the parent
HIP temporary file remained empty, and the profiler reported a ring-buffer
mapping error. Its partial archive cannot establish a complete request corpus,
complete graph replay, or clean shutdown.

Issue #3076 therefore needs a bounded, fail-closed worker-start profiler. It
must establish which process executed each GPU operation, prove that profiling
was active before runtime initialization, and finalize every required artifact
before the normal engine shutdown completes.

## Scope

In scope:

- one importable launch seam, one thin public diagnostic client, and versioned
  input contracts;
- one no-GPU worker-image discovery and seal before runtime-closure
  preparation;
- one launch owner for each arm, from profiler initialization through artifact
  finalization and production worker shutdown;
- production vLLM V2 and vllm.cpp process trees with their existing supervisor,
  worker, interprocess communication, graph, scheduler, and sampling behavior;
- the same pinned profiler build, configuration bytes, semantic trace
  categories, workload, and resource limits on both arms;
- a short readiness run before the full bounded corpus;
- fail-closed provenance, lifecycle, trace-completeness, scheduler-shape,
  graph-replay, and artifact-finalization checks;
- CPU red, green, and scratch-mutation tests before leased hardware work; and
- recovery from a host reboot using immutable source and evidence archives on
  the network-attached storage (NAS) volume.

Out of scope:

- late attachment to an initialized process;
- eager mode, vLLM V1, or a nonproduction execution denominator;
- `LD_PRELOAD`, an API shim, or another interposer that can change the
  numerical execution route;
- a global package or profiler install;
- a kernel optimization, default change, tolerance change, or performance
  attribution;
- treating instrumented time as accepted throughput or latency; and
- publishing a benchmark or closing the performance gap.

## Fixed inputs

The manifest must reject any different value instead of silently substituting
it:

| Input | Required value |
|---|---|
| vllm.cpp | `6e3cbfb940be89e28d1d71c264fd8c3a4e44afeb` |
| production vLLM | `e126687a9a828d513c01a07cd69f025f27d63280` |
| profiler harness repository | `8952c3c9e7712daf54521e5eb8a5b0a1ee9e1660` |
| nested rocprofiler SDK source | `97f5574fe2fdc7bef44fb01545347912ee9f1779` |
| profiler SDK | `1.1` |
| model | `Qwen/Qwen3-4B` |
| model revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| dtype | BF16 |
| prompts | the six raw prompts in the retained qualification manifest |
| sampling | greedy, 128 requested output tokens |
| concurrency | one and four, with concurrency four required for attribution |

The implementation must bind every prompt byte and the whole ordered schedule
by SHA256. It must also bind the tokenizer and every model shard before and
after each arm. The retained qualification manifest already records these
values and is the recovery source, not a license to accept a changed file.

The profiler configuration is one canonical byte sequence. Both arms must
record the same configuration SHA256 and the same tool and SDK-library hashes
and build IDs. The required semantic categories are HIP runtime APIs, kernel
dispatch, HSA APIs and AQL queue or dispatch activity, graph capture and replay,
and external correlation markers. The implementation must derive the exact
supported rocprofiler spellings from the pinned build and store them in the
manifest. A missing category, ignored option, or differing resolved
configuration fails the pair.

The retained SDK 1.1 tool library has SHA256
`478df9af09b74707652d9d5574ef37151ff1234d09c68843972c1409a505cdd0`
and build ID `0ca5ba0e4c583fcb8a8a2beb5698038cf3aff0e1`. The retained
`rocprofv3` file is a Python script. It is not that tool library.

The earlier environment recorded a different SDK library binary with SHA256
`40a5ecd8ca25dc3facb132b730066354812fe82d6889625d43371dbb0655b1ca`
and build ID `82dd8833b65c17523a3054f6b54da0e7a8831c82`. These values identify
built artifacts, not either source commit. The implementation must rebuild or
recover the pinned SDK binaries, verify these identities where the same
artifact is used, and record any newly built identity without treating it as
interchangeable with the retained binary.

## Source anchors

### Production vLLM

At vLLM commit `e126687a9a828d513c01a07cd69f025f27d63280`:

- `vllm/v1/worker/gpu/model_runner.py:1517` enters `execute_model`;
- `model_runner.py:1541-1554` derives the actual request and token shapes;
- `model_runner.py:1570-1580` dispatches compiled graphs and selects eager only
  for profiling or an explicit compiled-mode bypass;
- `model_runner.py:1591-1594` prepares persistent inputs and attention state;
- `model_runner.py:1664-1667` handles full-graph capture metadata;
- `model_runner.py:1731-1734` records actual batch size and the full-graph
  decision;
- `model_runner.py:1737-1743` enters full-graph replay;
- `model_runner.py:1766-1773` enters piecewise replay; and
- `vllm/v1/worker/gpu/cudagraph_utils.py:439-452` executes full-graph replay.

These are the production V2 anchors. The implementation must not replace them
with V1, `--enforce-eager`, or a synthetic model call.

The production worker bootstrap is also pinned. The SHA256 values below cover
the complete file bytes at vLLM commit `e126687a9a828d513c01a07cd69f025f27d63280`:

| Source | SHA256 | Decisive anchors |
|---|---|---|
| `vllm/v1/engine/utils.py` | `d09f86b107b3c63b62f40a6590036be5b711778ea997fc5a6af6de94fdd2ccd3` | lines 164 and 179-235 select the context, bind `EngineCoreProc.run_engine_core`, and call `proc.start()` |
| `vllm/v1/engine/core.py` | `f990e260cebf0826d08cfd30955c0c7a85b9af00504841b91759ca5bc7b036f6` | lines 23-95 import vLLM modules before `run_engine_core` at lines 1290-1311 |
| `vllm/utils/system_utils.py` | `837e6ff0ab524c06c048d6b42d4f35b77baba2e9d9e57837fe333ca0aaee1cfb` | lines 126-181 select and return the multiprocessing context |
| `vllm/envs.py` | `0e2dce375da631b54288a0f0ad783a46dadd9f19cac7c65724463031a99f9c9f` | lines 927-928 define `fork` as the default `VLLM_WORKER_MULTIPROC_METHOD` |

This order rules out a wrapper that starts inside `run_engine_core`. Python
imports `core.py` before that target runs. The accepted production-vLLM route
must retain the default `fork` start method and the exact target and keyword
arguments at `utils.py:190-195`. A changed start method fails manifest
validation and readiness.

### vllm.cpp

At vllm.cpp commit `6e3cbfb940be89e28d1d71c264fd8c3a4e44afeb`:

- `src/vllm/v1/worker/gpu/runner.cpp:3155` routes through
  `ModelRegistry::Forward`;
- `src/vllm/model_executor/models/qwen3_dense.cpp:85-103` enters the registered
  Qwen3 implementation and dispatches its graph at line 102;
- `src/vllm/model_executor/models/qwen3.cpp:1026-1040` opens the full-graph
  capture scope;
- `qwen3.cpp:1116` replays after capture; and
- `qwen3.cpp:1177-1186` resolves whether the static graph route is admissible.

The trace, not source inference, must establish the resolved vllm.cpp graph
mode and whether replay executed. A source candidate is not evidence that a
kernel or fallback ran.

### Profiler registration and attachment

The retained nested rocprofiler SDK source has these decisive anchors:

- `rocprofiler_register.cpp:335` resolves `rocprofiler_configure` with
  `dlsym(RTLD_DEFAULT, ...)`;
- `rocprofiler_register.cpp:342-363` accepts an existing provider before it
  considers loading another tool;
- `rocprofiler_register.cpp:790-840` creates the attachment proxy path only
  under its attachment and activation conditions, then propagates API tables
  to the active provider; and
- `sdk-attach.cpp` requests registration attach or detach but cannot recreate
  graph activity that occurred before its observation window.

The extracted 11,537-byte `include/rocprofiler-sdk/registration.h` has SHA256
`98eed78778685f6e01f9a29f9c415b8c0d5e0517476c5d8f84a978b37486ce0e`.
Its lines 104-115 define `rocprofiler_is_initialized`: status `1` means the SDK
completed its configuration scan. The pre-import bootstrap uses this public
query for its activation receipt.

The production-vLLM provider observation found both relevant GOT relocations
for `rocprofiler_configure` bound to the same symbol in `libtorch_cpu.so`. Its
status is `OBSERVED_NOT_COMPATIBILITY_PROOF`, and it is not a trace. Together
with the SDK source, it is enough to reject late attach as the authority for
already captured graphs.

## Evidence retained for recovery

All paths below are under
`/mnt/nas_share/rc/strix-four-engine-3053.X94a3J`. The implementation must
verify a retained file before using it and must copy no large archive to the
shared checkout.

| Evidence | SHA256 | Meaning |
|---|---|---|
| `qualification-manifest-12.json` | `4f0b0cbf0e06b7917405edc22bf852a31b037fce19361bde596841afda3c3f10` | pinned workload, model files, runtime, and built identities |
| `profiler-8952c3c9-source.tar` | `e4b6f98e097356e79848ec8640d980a8699f4ce0c2da36d6d740bfb88cc138a7` | profiler-harness repository source at `8952c3c9e7712daf54521e5eb8a5b0a1ee9e1660` |
| `vllmcpp-6e3cbfb-source.tar` | `df4ab94e5670ff7f7de2fe6427107153616643dbda9996b311398c91cfbb9fda` | vllm.cpp source pin |
| `oracle-c8d019447-source.tar` | `fa3ca302acb5eda709cfa567dd957411bee6a4a3e00f23632641e6130eb17f35` | retained vLLM oracle harness source |
| `got-provider-observation.jmEF9j/got-provider.json` | `48a8f328c3f1066464cddb4a64b0bd44a56b947592196e629f3cfc32594e5785` | Torch provider observation, not a trace |
| `vllm-matched-trace-06.log` | `9140e80cd1934b700b7d982a9d33b931f1a04cd2fd8aa008f795a2e71616938d` | process-start attempt and finalization failure |
| `vllm-trace06-preserve-07.log` | `dc5c732b9c69fd6d3bf31ace3735f07df673e3665f82aaac2a922c1b853f1d94` | preserved sizes and failure evidence |
| `vllm-trace06-partial-07.tar` | `fd25b19fbd02fc805e31c25a2ce8f1e21f6e1ad9b6135702d484b82cd4c44afc` | partial diagnostic output, never trace authority |
| `systemlibs3108.6xsEkD/Packages.gz` | `ca9ce1e681e736592a8dc8a7309a2ef5e0a71b7152da450de4dc672a3ce62e6e` | 61,300-byte package index that binds the two missing ROCm package payloads |

The harness at `8952c3c9e7712daf54521e5eb8a5b0a1ee9e1660` checked out the ROCm
monorepo and built `projects/rocprofiler-sdk` from nested source pin
`97f5574fe2fdc7bef44fb01545347912ee9f1779`. Historical build logs record
that relationship. The archive SHA256 binds the harness archive bytes, not the
nested SDK source or a built binary.

The retained evidence supplies no immutable worker-image authority. A complete
key and value-path inspection of `qualification-manifest-12.json` found no
image, container, root-filesystem, operating-system, boot, or dpkg identity.
Its `audit` field binds `audit-real-f99f5d3a.json` at SHA256
`4ab481289d457cf7a4f6f6073ff3430a0b5a92e2ff87b24dfc719b548c886656`.
That file is a model-tensor audit. It contains no worker identity. A scan of
the retained JSON receipts found no controller-authorized image digest or
worker-image receipt. A current worker cannot supply this missing authority
for itself.

The separate retained SDK source anchor files include
`sdk-attachment-source.TktBaK/rocprofiler_register.cpp` with SHA256
`a54e43b6546c006b635b263012b7d90abbaf18f64d022dff1a8a9adafc255f0e`
and `sdk-attachment-source.TktBaK/sdk-attach.cpp` with SHA256
`df9e1241170289b2753b1bab4f3ab0c81f8038df4314ccd8b5212ed62fd7b0b6`.
The historical patched controller was only `BUILT_NOT_HARDWARE_VALIDATED` and
does not become part of this design.

### Executable profiler recovery

The harness archive is not SDK source and is never an SDK recovery input. The
selected recovery route extracts Debian packages into a fresh worker-local
prefix. It never invokes `dpkg -i`, `apt install`, or writes under the host
`/opt`.

Three package payloads already exist under
`systemlibs3108.6xsEkD/`:

| Package | Bytes | SHA256 | Package metadata |
|---|---:|---|---|
| `rocprofiler-sdk_1.1.0-93~24.04_amd64.deb` | 5,680,890 | `39270239e68660cd025d3c9e84a04478696d9737f69012cd350c380aad911c30` | `rocprofiler-sdk`, `1.1.0-93~24.04`, `amd64`, depends on `rocm-core`, `rocprofiler-sdk-roctx (>= 1.1.0)`, and `rocprofiler-sdk-rocpd (>= 1.1.0)` |
| `rocprofiler-sdk-rocpd_1.1.0-93~24.04_amd64.deb` | 4,229,746 | `313873a14f76dde8f4ca2aa7fed8eae68240dc6ae446537dd3b363180c38095d` | `rocprofiler-sdk-rocpd`, `1.1.0-93~24.04`, `amd64`, depends on `rocm-core` |
| `rocprofiler-sdk-roctx_1.1.0-93~24.04_amd64.deb` | 258,610 | `4c45f467341b14fc1e1db3c9dc2475d7e880650df8ec71db46b461d7b12ffb91` | `rocprofiler-sdk-roctx`, `1.1.0-93~24.04`, `amd64`, depends on `rocm-core` and `rocprofiler-register` |

The retained package index binds two missing ROCm dependency payloads. The
Ubuntu Noble security index binds the third acquisition by its exact pool
filename. The recovery contract does not resolve a floating package version.

| Package | Bytes | SHA256 | Exact repository path |
|---|---:|---|---|
| `rocm-core_7.2.4.70204-93~24.04_amd64.deb` | 32,692 | `dfe0d173da998a669921faf08351a01add377cd4d1b471865f921a03484974b6` | `https://repo.radeon.com/rocm/apt/7.2.4/pool/main/r/rocm-core/rocm-core_7.2.4.70204-93~24.04_amd64.deb` |
| `rocprofiler-register_0.6.0.70204-93~24.04_amd64.deb` | 242,888 | `3b13874e567fa40b6eaf9f8d572d5c9a4ac780eb37496b31565755463a378295` | `https://repo.radeon.com/rocm/apt/7.2.4/pool/main/r/rocprofiler-register/rocprofiler-register_0.6.0.70204-93~24.04_amd64.deb` |
| `libsqlite3-0_3.45.1-1ubuntu2.8_amd64.deb` | 701,602 | `b1190bb72359f5fcc47406aa46065eaf4f1ca208085c51224a52b04bedc0b4bb` | `https://security.ubuntu.com/ubuntu/pool/main/s/sqlite3/libsqlite3-0_3.45.1-1ubuntu2.8_amd64.deb` |

Before a hardware lease, an authorized network-capable job must fetch those
three files into a new NAS staging directory. It uses these argument arrays and
never uses a shell:

```text
["/usr/bin/curl", "--fail", "--location", "--proto", "=https",
 "--tlsv1.2", "--output", "<stage>/rocm-core_7.2.4.70204-93~24.04_amd64.deb",
 "https://repo.radeon.com/rocm/apt/7.2.4/pool/main/r/rocm-core/rocm-core_7.2.4.70204-93~24.04_amd64.deb"]
["/usr/bin/curl", "--fail", "--location", "--proto", "=https",
 "--tlsv1.2", "--output",
 "<stage>/rocprofiler-register_0.6.0.70204-93~24.04_amd64.deb",
 "https://repo.radeon.com/rocm/apt/7.2.4/pool/main/r/rocprofiler-register/rocprofiler-register_0.6.0.70204-93~24.04_amd64.deb"]
["/usr/bin/curl", "--fail", "--location", "--proto", "=https",
 "--tlsv1.2", "--output",
 "<stage>/libsqlite3-0_3.45.1-1ubuntu2.8_amd64.deb",
 "https://security.ubuntu.com/ubuntu/pool/main/s/sqlite3/libsqlite3-0_3.45.1-1ubuntu2.8_amd64.deb"]
```

The job verifies all three byte counts and SHA256 values. It then atomically
moves the three files and one hash receipt into `systemlibs3108.6xsEkD/`. The
receipt binds every acquired filename, byte count, and SHA256 value. Until that
receipt exists, profiler recovery is `PENDING` and no readiness lease starts.
This is the only network acquisition in the recovery contract.

For every run, create `<run>/sdk-root` under a fresh `/tmp` directory. Verify
all six package files. For each package, run
`["/usr/bin/dpkg-deb", "-f", "<package>", "Package", "Version",
"Architecture", "Depends"]` and compare the exact fields earlier in this
section. Extract each package with
`["/usr/bin/dpkg-deb", "-x", "<package>", "<run>/sdk-root"]`.
Reject an absolute or symlinked prefix and reject any path that escapes it.

The extracted prefix must contain these identities:

| Relative path | Bytes | SHA256 | Build ID |
|---|---:|---|---|
| `opt/rocm-7.2.4/bin/rocprofv3` | 62,506 | `195ff5e6faf48a3abbc6f4db9f69dd598fe71fa9ff695ba2556d65af636fdc48` | n/a, Python script |
| `opt/rocm-7.2.4/lib/librocprofiler-sdk.so.1.1.0` | 8,314,944 | `40cf6fefffa5e9e8da249dbc1ce6feab0bb2613438caf37b0224d7fce09241e1` | `3b9c4332f65be22417cec9d7ba6bad2eb7e6e039` |
| `opt/rocm-7.2.4/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so.1.1.0` | 5,435,848 | `478df9af09b74707652d9d5574ef37151ff1234d09c68843972c1409a505cdd0` | `0ca5ba0e4c583fcb8a8a2beb5698038cf3aff0e1` |
| `opt/rocm-7.2.4/lib/librocprofiler-sdk-rocpd.so.1.1.0` | 493,128 | `000b898ef15ef5a5de6f4c61f11064e817883a02ddaa5610223fa394ff15cca0` | `c90598ff578bfcecba55e40d4e6291408d92843a` |
| `opt/rocm-7.2.4/lib/librocprofiler-sdk-roctx.so.1.1.0` | 456,232 | `1d99a44a8c24370dbbabbc1b68b6b9606db53c5662dffc2dac31948a377b0ca9` | `626cd91b7d6cacc633cb874faae0edda01eb967d` |
| `usr/lib/x86_64-linux-gnu/libsqlite3.so.0.8.6` | 1,468,440 | `85265a9d4afca6f4b325ceb078b669c754fb881abed4cafe91ccebe9d625d975` | `5701975a7ab1644d59e6b20df0257e183eafa78e` |

`libsqlite3-0` has version `3.45.1-1ubuntu2.8`, architecture `amd64`, and
dependency `libc6 (>= 2.38)`. Its shared object has SONAME
`libsqlite3.so.0`. The relative symlink
`usr/lib/x86_64-linux-gnu/libsqlite3.so.0` must target
`libsqlite3.so.0.8.6`.

### Recursive runtime-closure finding

The 19 September 2026 CPU inspection extracted all six packages into one local
prefix. A clean Python process loaded the SDK and tool with `ctypes.CDLL` and
`RTLD_GLOBAL`. The process printed `CDLL_LOAD_OK`, but that result did not prove
a reproducible dependency closure.

The final review evidence has SHA256
`91d16a335e113686e8661ce4ea94411e2c1d91009a846fc5f5f8b910eab23d2a`.
Its recursive audit reports `UNBOUND_COUNT=8`. The unbound set contains
`libm.so.6`, `libc.so.6`, the dynamic loader, `libdl.so.2`, `liblzma.so.5`,
`libbz2.so.1`, `libpthread.so.0`, and `librt.so.1`. The review host's libc,
libm, and dynamic-loader hashes differ from the retained Strix hashes. The
qualification manifest is therefore not a complete runtime-closure manifest.

### Public worker-image discovery and seal

Runtime-closure preparation cannot discover the image and then authorize its
own discovery. The implementation first adds the importable
`tools.bench.strix_worker_profile.worker_image.discover_and_seal` callable.
Its contract identifier is `vllm.cpp/worker-image-discovery/v1`.

The public signature is
`discover_and_seal(request: DiscoverWorkerImageRequest) -> DiscoverWorkerImageResult`.
`DiscoverWorkerImageRequest` is a closed tagged union. It has these two request
types and no untyped dictionary form:

- `DiscoverWorkerImageBootstrapRequest` has schema
  `vllm.cpp/worker-image-discovery-bootstrap-request/v1`. It contains the
  qualification-manifest bytes and SHA256, the literal expected device
  `strix:gpu0`, the fixed ordered 12-key list, and a new output directory.
  It contains no baseline, lease, boot, image, package, or library identity.
- `DiscoverWorkerImageReplayRequest` has schema
  `vllm.cpp/worker-image-discovery-replay-request/v1`. It contains every
  bootstrap field, the accepted profile-manifest bytes and SHA256, the prior
  candidate baseline-receipt bytes, its detached SHA256 value, and one
  `context_relation` value. The allowed values are
  `same-lease-same-boot`, `fresh-lease-same-boot`, and
  `fresh-lease-changed-boot`.

The bootstrap type contains exactly `schema`, `qualification_manifest_bytes`,
`qualification_manifest_sha256`, `expected_device`, `resolver_keys`, and
`output_directory`. The replay type adds exactly
`accepted_profile_manifest_bytes`, `accepted_profile_manifest_sha256`,
`baseline_receipt_bytes`, `baseline_receipt_sha256`, and `context_relation`.
Every `*_bytes` value is an immutable byte sequence. Every `*_sha256` value is
a typed 32-byte value. `resolver_keys` is a fixed 12-string tuple.
`output_directory` is an absolute nonsymlink path that must not exist.

The `--discovery-request` JSON file represents each byte sequence as padded
RFC 4648 base64 and each SHA256 as 64 lowercase hexadecimal characters. It
contains every matching typed field except `output_directory`. The thin client
adds only the separate `--output` value. It rejects a request-file output field
or a CLI field that duplicates any other request member. This encoding uses
the canonical JSON rules defined for the runtime envelope.

Both parsers reject an unknown field, duplicate JSON key, another key order,
another resolver-key set, or an identity for the current worker. Caller-supplied
receipt, hash, and manifest bytes are untrusted carriers. Their matching hashes
prove integrity only; they do not prove campaign provenance.

### Campaign baseline acceptance and authority

Bootstrap produces an unaccepted candidate only. A bootstrap receipt can never
authorize replay or preparation, even when its detached hash is valid. A
second bootstrap starts another candidate lineage and cannot continue this
campaign.

The operator promotes one candidate in a separate record-only change. That
change creates the tracked file
`.agents/campaign-baselines/BACKEND-GATE-ROCM-SGLANG-3076.json`. No shared
index lists campaign commitments. The file is absent in this design change,
so the real replay and hardware gate remains `PENDING`.

The commitment has schema and hash domain
`vllm.cpp/worker-image-campaign-baseline/v1`. It contains exactly these ten
members:

1. `schema`, with that literal value.
2. `row`, with the literal `BACKEND-GATE-ROCM-SGLANG`.
3. `issue`, with the literal `ISSUE-GH-3076`.
4. `campaign_id`, as one nonempty ASCII string selected during promotion.
5. `candidate_receipt_sha256`, for the exact bootstrap receipt bytes.
6. `qualification_manifest_sha256`, copied from that receipt.
7. `accepted_profile_manifest_sha256`, for the promoted strict manifest.
8. `runtime_envelope_sha256`, copied from that receipt.
9. `binding_fingerprints`, as the exact ordered 12-value array defined later.
10. `acceptance_state`, with the literal `reviewed`.

The commitment uses the envelope's normalized primitives and duplicate rules.
It is compact UTF-8 JSON with keys sorted by code point, shortest escapes and
integers, declared array order, and one terminal line feed. Its digest input is
the ASCII domain `vllm.cpp/worker-image-campaign-baseline/v1`, one NUL byte,
and those canonical file bytes. SHA256 produces exactly 64 lowercase
hexadecimal characters. Replay and preparation recompute this digest; no
detached caller-supplied value can replace it.

The operator copies these identities from one candidate and never derives them
from the current worker. The operator sets `acceptance_state=reviewed` during
promotion and commits the final file. That string is not self-authenticating.
A fresh reviewer compares the candidate, profile manifest, and commitment at
the immutable head before any replay or hardware gate. Git history is the
review authority; this design invents no controller signature.

The production loader resolves the repository root from the tracked
`tools/bench/strix_worker_profile/worker_image.py` module. It then loads only
the fixed path above. It accepts no commitment path, bytes, digest, or override
from a request, command-line argument, environment variable, or current-worker
observation. The loader requires a regular nonsymlink file whose bytes equal
the file at `HEAD` and whose schema, row, issue, and acceptance state match.
The operator must run from the independently reviewed acceptance head. A
missing, dirty, unreviewed, malformed, or mismatched commitment returns
`PENDING_CAMPAIGN_BASELINE` before discovery reads `RC_DEVICE`, `RC_JOB_ID`,
the boot ID, dpkg state, or a library file.

Replay first loads and validates this campaign commitment. It then requires
the caller-carried candidate receipt and accepted profile manifest to match
the two hashes in the commitment. It requires the receipt's qualification
manifest, envelope, and 12 fingerprints to match the commitment. Only after
these checks can replay observe the current host. The runtime caller cannot
substitute the commitment, the accepted baseline, or the current observation.

The accepted profile manifest's `worker_image_baseline` object has schema
`vllm.cpp/worker-image-baseline/v1`. It contains exactly `schema`,
`campaign_id`, `candidate_receipt_sha256`, `qualification_manifest_sha256`,
`runtime_envelope_sha256`, and `binding_fingerprints`. Each value must equal
the campaign commitment. The commitment binds the exact whole profile manifest
through `accepted_profile_manifest_sha256`, so a caller cannot replace another
manifest that repeats this nested object.

Implementation tests can inject a commitment loader that reads an immutable
temporary fixture. That test seam is not exported, is unavailable to the
public command, and is not production authority. A fixture with
`acceptance_state=reviewed` proves parsing and comparisons only.

The controller supplies lease authority outside either JSON request. The
callable reads `RC_DEVICE` and `RC_JOB_ID` from the process that `rc run`
started. It requires `RC_DEVICE=strix:gpu0` and a nonempty `RC_JOB_ID`.
The authoritative boot source is
`procfs:/proc/sys/kernel/random/boot_id`. Its bytes must match one lowercase
UUID plus one line feed. The receipt stores the 36-character UUID and the
literal source name. No request can supply the current lease, device, or boot.

The campaign commitment is the baseline authority for downstream phases. A
replay receipt and detached hash bind the current observation to that authority.
The runtime-relevant image envelope has schema and hash domain
`vllm.cpp/worker-image-runtime-envelope/v1`. It contains exactly these nine
top-level members:

1. `schema`, with that literal value.
2. `os_release`, a `ByteFileV1` for `/etc/os-release`.
3. `dpkg_status`, a `ByteFileV1` for `/var/lib/dpkg/status`.
4. `dpkg_architecture`, the normalized `dpkg --print-architecture` value.
5. `dpkg_executables`, the canonical two-entry `ByteFileV1` array for
   `/usr/bin/dpkg` and `/usr/bin/dpkg-query`, sorted by path.
6. `dpkg_packages`, the complete normalized package-row array.
7. `dpkg_metadata_files`, the complete normalized array of dpkg list files
   used by the 12 ownership records.
8. `qualification_records`, the exact eight qualification-bound bindings.
9. `live_bindings`, the exact 12 discovery bindings.

`ByteFileV1` contains exactly `path`, `byte_count`, `sha256`, and
`content_base64`. `path` is the absolute canonical path. `byte_count` is the
raw content length. `sha256` hashes the same raw bytes. `content_base64` is
padded RFC 4648 base64 with no whitespace. Decoding must reproduce the byte
count and SHA256. Raw file bytes never pass through a text normalization.

The implementation obtains `dpkg_architecture` from the exact argument array
`["/usr/bin/dpkg", "--print-architecture"]`. The output must be one nonempty
ASCII token plus one line feed. The envelope stores the token without the line
feed. The package query uses
`["/usr/bin/dpkg-query", "-W", "-f=${binary:Package}\\t${Version}\\t${Architecture}\\n"]`.
Each output row must contain three nonempty ASCII fields and one line feed.
The envelope represents each row as an object with exactly `package`,
`version`, and `architecture`. It sorts rows by those three UTF-8 byte strings
and rejects a duplicate tuple or duplicate package and architecture pair.

Each `dpkg_metadata_files` entry is a `ByteFileV1`. The array contains one
entry for each distinct canonical dpkg `.list` file named by a live binding.
It sorts entries by path and rejects duplicate paths. `dpkg_executables` uses
the same ordering and rejects a missing, extra, or duplicate executable path.
No other file under `/var/lib/dpkg/info` belongs to the envelope.

Each `qualification_records` entry is a `QualificationBindingV1`. It has
exactly `resolver_key`, `source_class`, `manifest_pointer`, `absolute_path`,
`canonical_path`, `symlink_chain`, `byte_count`, `sha256`, `gnu_build_id`,
`dt_soname`, and `dt_needed`.
`source_class` is `sealed-non-glibc`. `manifest_pointer` is the unique JSON
Pointer into the strict qualification manifest. The array contains exactly
`libamd_comgr.so.3`, `libdrm.so.2`, `libdrm_amdgpu.so.1`, `libdw.so.1`,
`libelf.so.1`, `libhsa-amd-aqlprofile64.so.1`, `libhsa-runtime64.so.1`, and
`libnuma.so.1`. It sorts entries by resolver key and rejects a duplicate key,
path, canonical path, or manifest pointer. The observed file must match every
manifest field before it enters the envelope.

Each `live_bindings` entry is a `WorkerFileBindingV1` with exactly these
members:

- `resolver_key`, `source_class`, `absolute_path`, and `canonical_path`;
- `symlink_chain`, as traversal-ordered objects containing exact `path` and
  relative `target` strings;
- `byte_count`, `sha256`, `gnu_build_id`, `dt_soname`, and `dt_needed`;
- `package`, `version`, `architecture`, `dpkg_query_s_row`, and
  `dpkg_query_w_row`;
- `dpkg_list_file`, a `ByteFileRefV1` with exactly `path` and `sha256`; and
- `provenance_kind`, `loader_input`, and `witness`.

The array uses the six `sealed-non-glibc` keys followed by the six
`host-glibc-witness` keys in the exact order listed later in this section.
It rejects another order or a duplicate resolver key, path, canonical path,
symlink-chain path, or package ownership row. `dt_needed` preserves ELF entry
order and rejects a duplicate needed name. `symlink_chain` preserves traversal
order and must end at `canonical_path`. The six glibc entries set
`loader_input=false` and `witness=true`. The other six entries use the inverse.

`WorkerFileBindingFingerprintV1` contains exactly `resolver_key` and `sha256`.
Its `resolver_key` equals the source binding's resolver key. Its `sha256` is
exactly 64 lowercase hexadecimal characters over the framed preimage defined
here.

The preimage object is the complete `WorkerFileBindingV1`. It contains exactly
these 19 members: `resolver_key`, `source_class`, `absolute_path`,
`canonical_path`, `symlink_chain`, `byte_count`, `sha256`, `gnu_build_id`,
`dt_soname`, `dt_needed`, `package`, `version`, `architecture`,
`dpkg_query_s_row`, `dpkg_query_w_row`, `dpkg_list_file`, `provenance_kind`,
`loader_input`, and `witness`. `symlink_chain` and `dpkg_list_file` retain the
closed member sets defined earlier. An omitted member is not equivalent to an
empty string, empty array, zero, `false`, or JSON `null`.

The fingerprint encoder applies the envelope's normalized string, integer,
Boolean, path, array, and duplicate rules. It emits compact UTF-8 JSON with
object keys sorted by code point, shortest required escapes, shortest unsigned
base-10 integers, the declared nested-array orders, and one terminal line feed.
It rejects duplicate object keys before typed decoding. The digest input is the
ASCII domain `vllm.cpp/worker-file-binding-fingerprint/v1`, one NUL byte, and
those canonical bytes. SHA256 produces the `sha256` member's lowercase value.

Every receipt, profile manifest, and campaign commitment uses the same ordered
12-element `WorkerFileBindingFingerprintV1` array. The order is the six
`sealed-non-glibc` keys followed by the six `host-glibc-witness` keys. Each
surface rejects a missing, extra, reordered, or duplicate resolver key and a
duplicate fingerprint digest. It recomputes each digest from the full binding
before comparison. No surface can use a path-only, hash-only, or independently
serialized fingerprint.

Each `ByteFileRefV1` must resolve to exactly one byte-identical
`dpkg_metadata_files` entry. An absent, ambiguous, or unreferenced metadata
entry fails. No nested representation can override a top-level byte identity.

All schema-controlled strings and discovered path, package, ELF, and dpkg-row
strings must be valid ASCII. The parser rejects a byte-order mark, carriage
return, NUL, invalid UTF-8, non-ASCII code point, float, negative integer,
boolean in an integer field, or integer outside unsigned 64-bit range. JSON
integers use the shortest base-10 form. JSON strings preserve exact code units;
no Unicode normalization or case folding occurs.

Canonical envelope bytes are UTF-8 JSON with object keys sorted by code point,
compact separators, the declared array orders, and one terminal line feed.
The encoder emits only the shortest required JSON escapes. It rejects duplicate
object keys before decoding into a typed value. The digest input is the ASCII
domain string `vllm.cpp/worker-image-runtime-envelope/v1`, one NUL byte, and
the canonical envelope bytes. SHA256 produces the envelope digest, encoded as
exactly 64 lowercase hexadecimal characters. Structured JSON and base64 avoid
unframed concatenation. The typed member list makes omission distinguishable
from an empty value.

The envelope excludes `RC_JOB_ID`, `RC_DEVICE`, boot ID, host name, mount
label, loader-cache entry, timestamps, and the output path. The discovery
receipt binds controller-issued lease and device authority, kernel boot
authority, the canonical envelope bytes, and its digest as separate members.
Controller authority can select where discovery runs, but it cannot change the
derived envelope. The digest makes no claim about unrelated image files.

The discovery phase runs through `rc run strix:gpu0`, never through SSH. It
requires `RC_DEVICE=strix:gpu0` and a nonempty `RC_JOB_ID`. It initializes no
GPU runtime and makes no device call. It imports no engine, Torch, ROCm runtime,
or profiler library. It records `/proc/self/maps` before and after discovery.
It rejects a new HSA, HIP, profiler, engine, or GPU mapping.

The phase reads `/etc/os-release`, `/var/lib/dpkg/status`, the dpkg
architecture, and the sorted `dpkg-query -W` rows. It binds each byte sequence,
byte count, and SHA256. It also binds both dpkg executables and every dpkg
metadata file that supplies ownership for a selected file.

The exact discovery set contains these six `sealed-non-glibc` keys:

- `libbz2.so.1`;
- `libgcc_s.so.1`;
- `liblzma.so.5`;
- `libstdc++.so.6`;
- `libz.so.1`; and
- `libzstd.so.1`.

It also contains these six `host-glibc-witness` keys:

- `ld-linux-x86-64.so.2`;
- `libc.so.6`;
- `libdl.so.2`;
- `libm.so.6`;
- `libpthread.so.0`; and
- `librt.so.1`.

The phase discovers exactly 12 records. It enumerates only dpkg-owned files
whose basename or ELF `DT_SONAME` matches one declared key. It resolves each
symlink without crossing the worker root. Each key must select exactly one
canonical file. A missing key, unowned file, multiple owner, or two distinct
canonical candidates fails. Search order, the live loader cache, and a process
default cannot select a file.

Each selected record binds the resolver key, source class, absolute path,
canonical path, complete relative symlink chain, bytes, SHA256, GNU build ID,
ELF `DT_SONAME`, and ordered `DT_NEEDED` list. It also binds the owning package,
version, architecture, exact `dpkg-query -S` row, exact `dpkg-query -W` row,
and the byte identity of the package's dpkg list file. Missing data fails. The
six glibc records remain witnesses and never become loader inputs.

The deterministic receipt has schema
`vllm.cpp/worker-image-discovery-receipt/v1`. It uses the same canonical JSON
rules. It contains exactly these 15 members: `schema`, `request_schema`,
`request_mode`, `qualification_manifest_sha256`,
`campaign_baseline_commitment_sha256`, `baseline_receipt_sha256`,
`context_relation`, `controller_lease_job`, `device`, `boot_id`, `boot_source`,
`runtime_envelope`, `runtime_envelope_sha256`, `binding_fingerprints`, and
`baseline_comparison`.

`baseline_comparison` contains exactly `status`, `envelope_sha256_match`, and
`binding_match_count`. Bootstrap uses `request_mode=bootstrap`,
`context_relation=initial-baseline`, and `status=not-applicable`. It sets both
SHA256 lineage members, `envelope_sha256_match`, and `binding_match_count` to
JSON `null`. Replay uses `request_mode=replay`, the requested context relation,
and `status=matched`. It records the canonical campaign-commitment digest, the
candidate receipt SHA256, `envelope_sha256_match=true`, and
`binding_match_count=12`. Another value or member set fails parsing.

The receipt carries the canonical runtime envelope and the ordered 12
`WorkerFileBindingFingerprintV1` values. Its qualification-manifest hash must
equal the request. A replay receipt's lineage and comparison members must match
the loaded campaign commitment and the recomputed current observation. The
phase writes
`worker-image-discovery-receipt.json` and
`worker-image-discovery-receipt.json.sha256` in a fresh staging directory.
The receipt ends in one line feed. The detached file is exactly 65 ASCII bytes:
64 lowercase hexadecimal SHA256 characters over the exact receipt bytes, then
one line feed. It has no filename, spaces, carriage return, or byte-order mark.

Discovery owns publication. It fsyncs both files and the staging directory,
publishes the directory with one same-filesystem rename, and fsyncs the parent.
It refuses overwrite. Failure injection before either file fsync, the staging
directory fsync, rename, or parent fsync must leave no usable final directory.
The receipt records content and lineage, not a claim about those write calls.

The public discovery command is:

```text
python3 tools/bench/strix_worker_profile/worker.py \
  --phase discover-worker-image \
  --discovery-request <discovery-request.json> \
  --output <new-directory>
```

This phase rejects `--engine` and every closure, readiness, or trace argument.
The command is a thin client. It parses bytes, constructs one request, invokes
the importable callable exactly once, and maps the result status to its exit
status. It cannot implement another discovery or publication path.

A receipt applies to one recorded lease job and boot. A bootstrap receipt is a
candidate and is never preparation authority. Preparation consumes only the
newest replay receipt from that same lease and boot. Replay observes a new
current envelope only after it validates the campaign commitment. It compares
the new digest and all 12 binding fingerprints with that commitment. It never
copies a current value into a baseline field.

The context relation has these exact semantics:

- `same-lease-same-boot` requires the current job, device, and kernel boot ID
  to equal the baseline authority fields.
- `fresh-lease-same-boot` requires a different nonempty job, the same device,
  and the same kernel boot ID.
- `fresh-lease-changed-boot` requires a different nonempty job, the same
  device, and a different kernel boot ID.

Only the permitted job and boot fields can differ. Every relation requires an
identical envelope digest and an explicit field-by-field match for all 12 live
bindings. A changed boot does not authorize image drift. An unexpected lease
or boot relation returns `CONTEXT_RELATION_MISMATCH`. An envelope-member change
returns `IMAGE_DRIFT`. A 12-binding change also returns `FILE_DRIFT`, even when
another comparison already reported `IMAGE_DRIFT`. The result records all
applicable failures and publishes no sealed receipt.

Discovery returns one of these fail-closed states. `SEALED_CANDIDATE` names a
complete bootstrap receipt and detached hash, but it grants no downstream
authority. `SEALED` names a replay receipt that matches the trusted campaign
commitment. `PENDING_CAMPAIGN_BASELINE` means that the fixed tracked commitment
is missing, dirty, unreviewed, malformed, or mismatched.
`PENDING_CONTROLLER_AUTHORITY` means that controller lease or device authority
is absent. `PENDING_IMAGE_AUTHORITY` applies only to a bootstrap that cannot
derive a complete envelope. `INVALID_BASELINE` means that replay's
caller-carried candidate receipt, detached hash, or accepted manifest does not
match the campaign commitment. `CONTEXT_RELATION_MISMATCH`, `IMAGE_DRIFT`, and
`FILE_DRIFT` have the meanings defined earlier. `FAILED_PUBLICATION` preserves
a bounded failure result outside the final name. Only replay status `SEALED`
reaches runtime-closure preparation.

### Public runtime-closure preparation

The implementation adds the importable
`tools.bench.strix_worker_profile.runtime_closure.prepare` callable. Its
contract identifier is `vllm.cpp/runtime-loader-closure/v1`. The callable owns
closure discovery, validation, copying, serialization, archive creation, and
atomic publication. No hidden script or launch-client branch can implement a
second closure algorithm.

Its public signature is
`prepare(request: RuntimeClosureRequest) -> RuntimeClosureResult`.
`RuntimeClosureRequest` contains exactly `profile_manifest_bytes`,
`worker_image_receipt_bytes`, `worker_image_receipt_sha256`, and
`output_directory`. The SHA256 field is a typed 32-byte value, not a path or
unchecked string. The request constructor recomputes SHA256 over the exact
receipt bytes and requires equality before `prepare()` can run.
`RuntimeClosureResult` contains the contract identifier, status, output path,
manifest, archive, receipt identities, fixed-point file and edge counts, total
bytes, and `UNBOUND_COUNT`. The implementation owns these types and their
strict JSON encodings in the same module as the callable.

The public preparation command is:

```text
python3 tools/bench/strix_worker_profile/worker.py \
  --phase prepare-runtime-closure \
  --manifest <manifest.json> \
  --worker-image-receipt <worker-image-discovery-receipt.json> \
  --worker-image-receipt-sha256 \
    <worker-image-discovery-receipt.json.sha256> \
  --output <new-directory>
```

The hash flag is mandatory for preparation. Its file must contain exactly 65
ASCII bytes: 64 lowercase hexadecimal characters and one line feed. The thin
client reads both regular nonsymlink files as bytes and passes those bytes to
the runtime-closure module's request decoder. The decoder owns exact-length,
case, line-feed, whitespace, byte-order-mark, carriage-return, typed-hash, and
receipt-hash validation. The client contains no second implementation of those
checks. A missing file, crossed pair, or swapped path fails request construction
before `prepare()` runs. A valid pair reaches `prepare()` exactly once.

The preparation phase rejects `--engine`. It creates one engine-independent
closure for both later arms. The existing profiled-process launch seam consumes
the sealed result. It does not discover or acquire a library itself.

The operator later runs this phase through `rc run strix:gpu0`, never through
SSH. `prepare()` first loads the fixed tracked campaign commitment through the
same non-overridable production loader that replay uses. The request accepts no
commitment member. A missing, dirty, unreviewed, malformed, or mismatched
commitment returns `PENDING_CAMPAIGN_BASELINE` before `prepare()` reads lease,
boot, package, or library state.

The campaign commitment supplies baseline provenance. The replay receipt and
detached hash supply the integrity of the current observation. Preparation
requires `request_mode=replay`, `status=matched`, and `SEALED` discovery status.
It rejects every bootstrap receipt, including a byte-valid
`SEALED_CANDIDATE`. It must not rediscover a path, package, image, or binding
from the live host. After the authority checks, it verifies the lease job,
device, boot ID, and image digest against the receipt. It then
byte-matches `/etc/os-release`, the dpkg bindings, the qualification-bound
eight-file subset, and all 12 sealed paths against that receipt. Verification
cannot change a selected path or provenance record. A mismatch fails before
closure discovery. Preparation and discovery must run in the same boot.

The profile manifest's exact bytes must hash to
`accepted_profile_manifest_sha256` in the campaign commitment. Its
`worker_image_baseline` object must equal the commitment's candidate receipt,
envelope, and 12 fingerprint values. The replay receipt must name the canonical
commitment digest and candidate receipt hash. It must also record the successful
12-member comparison. Preparation rejects any other lineage before it reads a
package payload.

Preparation returns `PENDING_WORKER_IMAGE_RECEIPT` when the receipt or detached
hash is missing. It returns `FAILED_WORKER_IMAGE_BINDING` for an invalid hash,
wrong job or boot, changed image, changed sealed file, or campaign-lineage
mismatch. Neither state can reach package extraction or the fixed-point walk.
Preparation enforces observable completeness and integrity of the final
receipt directory. It cannot determine whether those bytes reached the final
directory through an atomic rename. Only discovery owns and tests that write
mechanic. Preparation never converts a current-host observation into a new
receipt. It verifies every qualification-bound file before it resolves one
dependency.

The phase must not initialize HSA, HIP, or a GPU. It imports no engine, Torch,
ROCm runtime, or profiler library. It records `/proc/self/maps` before and after
preparation. It rejects a new HSA, HIP, profiler, engine, or GPU mapping. It
also rejects GPU device access.

The phase has a 10-minute wall timeout. It permits at most 64 ELF files and
512 MiB of canonical ELF bytes. The deterministic archive must also be at most
512 MiB. Check the NAS capacity before work starts. Reject a special file,
absolute symlink, path escape, hard link, socket, device, or FIFO.
The complete root-inclusive review walk contains 25 ELF file records, 96
dependency edges, and 192,846,932 canonical ELF bytes. The earlier 24-file,
187,411,084-byte audit started from the tool library's dependencies but omitted
the tool-library root itself. The 512 MiB limit therefore leaves 39 file slots
and 344,023,980 bytes of raw-ELF headroom. The archive has the same 512 MiB
ceiling, so tar metadata must fit inside that separately enforced bound. These
limits provide bounded headroom without authorizing closure growth.

### Fixed-point closure algorithm

Start from these exact extracted roots:

- `opt/rocm-7.2.4/lib/librocprofiler-sdk.so.1.1.0`;
- `opt/rocm-7.2.4/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so.1.1.0`;
- `opt/rocm-7.2.4/lib/librocprofiler-register.so.0.6.0`; and
- `usr/lib/x86_64-linux-gnu/libsqlite3.so.0.8.6`.

The manifest's `soname_bindings` object is a closed map. Search-root order is
not a resolver and cannot select a file. Each root and every `DT_NEEDED` name
must select exactly one source class and one individually named source file.
The only resolver keys and source classes are:

| Source category | Resolver keys | Required source |
|---|---|---|
| `six-package` | `librocprofiler-sdk.so.1`, root-only `librocprofiler-sdk-tool.so.1`, `librocprofiler-sdk-rocpd.so.1`, `librocprofiler-register.so.0`, alias `librocprofiler-register.so`, and `libsqlite3.so.0` | The verified DEB payload under `sdk-root`. |
| `sealed-non-glibc` | `libamd_comgr.so.3`, `libdrm.so.2`, `libdrm_amdgpu.so.1`, `libdw.so.1`, `libelf.so.1`, `libhsa-amd-aqlprofile64.so.1`, `libhsa-runtime64.so.1`, `libnuma.so.1`, `libbz2.so.1`, `libgcc_s.so.1`, `liblzma.so.5`, `libstdc++.so.6`, `libz.so.1`, and `libzstd.so.1` | The individually bound worker-image file. |
| `host-glibc-witness` | `ld-linux-x86-64.so.2`, `libc.so.6`, `libdl.so.2`, `libm.so.6`, `libpthread.so.0`, and `librt.so.1` | The witness-only live host component, byte-matched at launch. |

This map selects 5 canonical `six-package` files, 14
`sealed-non-glibc` files, and 6 `host-glibc-witness` files. The two register
keys select one canonical package file. Discovery can supply identities only
for the final 6 non-glibc keys and 6 glibc keys. It cannot change the 5/14/6
source-class map.

Each `six-package` key resolves only from its verified DEB payload under
`sdk-root`. Both register keys select the same canonical package file. Each
`sealed-non-glibc` record binds its absolute source path, symlink chain, bytes,
SHA256, GNU build ID, provenance kind, and package or qualification identity.
Each glibc record binds the exact live path and identity during preparation.
The launch must byte-match that live component. The archived witness is never
a loader input.

The first class has precedence for its declared keys even when the pinned image
contains another file with the same name. The second class has precedence for
its declared keys, and the third is the only admissible source for its six
keys. A `DT_NEEDED` name absent from this table is undeclared and fails. The
resolver never searches another class, falls back to a process default, or
substitutes a same-named file. Within the selected source class, a missing path
or two distinct canonical files in the selected source class for one key
fails. A same-named file in an unselected class is recorded as a shadowing
candidate but cannot create ambiguity or become a fallback.

The eight ROCm-adjacent `sealed-non-glibc` files selected from the production
vLLM environment (`libamd_comgr`, both DRM libraries, `libdw`, `libelf`, both
HSA libraries, and `libnuma`) use the exact path and SHA256 record in the
retained qualification manifest as their source provenance. The six selected
system non-glibc files and six glibc witnesses use `live-dpkg` ownership:
`dpkg-query -S` must return exactly one package for the canonical live path,
and `dpkg-query -W` binds its package, version, and architecture. An unowned or
multiply owned live path fails.

Files extracted below `sdk-root` do not use the live dpkg database for
ownership. Their `DEB-payload provenance` is the verified package filename and
SHA256, exact archive member path, and the package, version, and architecture
read from that DEB. The member must occur exactly once in the verified payload.
This distinction prevents an extracted path from being rejected merely because
it was never installed, while still making every selected worker-image file
and every package payload independently attributable.

Use this deterministic walk:

1. Validate the closed binding map, its exact source paths and identities, its
   provenance records, and the one allowed source class for every declared key.
2. Verify each root's canonical path, byte count, SHA256, and GNU build ID plus
   its DEB-payload provenance. Put the roots in a queue sorted by canonical
   relative path.
3. Read each dequeued ELF's `DT_SONAME` and ordered `DT_NEEDED` entries.
4. Look up each needed name in the closed map. Validate only its selected file
   in its selected class. Never search another class or a default loader path.
5. Reject an undeclared or missing name, an identity mismatch, or two distinct
   canonical files in the selected source class. Validate every alias and
   symlink chain against the selected canonical file.
6. Record every dependency edge and enqueue each new canonical file once.
7. Enforce the file-count and byte limits before each insertion.
8. Repeat until the sorted queue is empty and the resolved set reaches a fixed point.
9. Assemble the six-package root, copied non-glibc root, and glibc witness set,
   then run the same closed-map walk again. Require the identical selected
   canonical-file set and identical ordered edge set, with identical identities
   and source classes.

Both walks must report 25 files, 96 edges, 192,846,932 bytes, and
`UNBOUND_COUNT=0` for the reviewed fixture. Preparation on the pinned image
also records its independently recomputed values. Any drift from the strict
request fails. Reject a duplicate manifest record, duplicate archive path,
unresolved or undeclared edge, path escape, or missing provenance. Reject a
missing hash, missing build ID, or second-walk change to a selected file, edge,
identity, or source class.

Each record contains its SONAME, canonical path, symlink chain, byte count,
SHA256, GNU build ID, and ordered `DT_NEEDED` list. It also contains the owning
package, version, architecture, and source category.
The allowed source categories are `six-package`, `sealed-non-glibc`, and
`host-glibc-witness`. It also records the resolver key, acquisition source path,
provenance kind, and whether the file is a loader input or a witness. Apply
DEB-payload provenance to `six-package` records, qualification-manifest
provenance to the eight named vLLM-environment files, and live-dpkg ownership
to the selected live worker-image records. No ownership mechanism can silently
substitute for another.

### Sealed closure and glibc safety

Copy every resolved file outside the six-package root into
`runtime-closure-root`. Preserve each relative symlink chain inside that root.
Copy the glibc-family files as identity witnesses, but never use those copies
as loader inputs. Normalize directories to mode `0755` and regular ELF files to
mode `0555`. Preserve no timestamp, owner, group, extended attribute, or other
host metadata.

Serialize the closure manifest as UTF-8 JSON with sorted keys, compact
separators, and one terminal newline. Sort file records by source category,
SONAME, and canonical path. Sort edges by requester path, needed name, and
resolved path. Build a deterministic `ustar` archive with lexical path order,
numeric owner and group zero, modification time zero, and normalized modes.

The phase writes the manifest, archive, and hash receipt into a fresh staging
directory on NAS. The receipt binds their names, byte counts, SHA256 values,
schema identifiers, file count, and total ELF bytes. After fsyncing all three
files and the staging directory, the phase atomically publishes them with one
same-filesystem rename and fsyncs the parent directory. It refuses overwrite
and retains a bounded failure result outside the final name.

The closure receipt also binds the canonical campaign-commitment SHA256, the
candidate receipt SHA256, the accepted profile-manifest SHA256, and the replay
receipt SHA256. These four values provide the closure lineage without changing
the immutable campaign commitment after preparation.

The sealed archive is the only source for a non-six-package dependency after a
reboot. The byte-matched host glibc set is the only exception, and it is an
execution requirement rather than an acquisition fallback. No live-library
fallback is permitted. A `PENDING` result without a bounded archive is
insufficient. It cannot authorize readiness.

Do not inject an alternate libc or dynamic loader into the running Python
process. Before bootstrap, require the target interpreter and host libc, libm,
the dynamic loader, libdl, libpthread, and librt to byte-match their sealed
records. Compare paths, symlink chains, bytes, SHA256 values, build IDs, and
package identities. A mismatch stops before `sitecustomize` or user code.

Extract non-glibc dependencies into a fresh `runtime-closure-root`. Add only its
non-glibc directories and the six-package directories to the local loader
search path. Exclude every archived `host-glibc-witness` path. Reject a mapped
closure dependency from any live system path.

After a reboot, validate the receipt, manifest, archive, and every extracted
file. Rerun the fixed-point walk against only the six-package root, the closure
root, and the byte-matched host glibc set. Require the same file and edge sets,
the same hashes, no closure growth, and `UNBOUND_COUNT=0` before any CDLL load
or bootstrap.

The package SDK library does not match the earlier 8,133,513-byte SDK library
with build ID `82dd8833b65c17523a3054f6b54da0e7a8831c82`. Do not substitute one
for the other. The recovered package set becomes one new pinned profiler
identity only after dependency closure and readiness pass. Both trace arms
must use that same identity.

After a host reboot, reconstruct the profiler only from the six packages and
the sealed runtime-closure archive. Recover the engines from their verified
archives and the pinned model cache on NAS. Use a fresh worker-local directory
under `/tmp` and a project virtual environment under `/workspace`. Copy only
finalized, bounded evidence back to a new NAS directory. Do not use the harness
archive as SDK source, use a global install, reuse an unverified build, or
duplicate the model tree.

## Alternatives and decision

### Rejected: attach after the engine starts

The attach safety preflight proved process identity and avoided mutation. It
did not prove trace completeness. Torch/Kineto can register before attachment,
and the attachment proxy path is conditional. A trace that misses previously
captured HIP graphs fails the required graph-replay evidence. More retries
cannot change this mechanism.

### Rejected: profile only a generic parent process

Wrapping a supervisor is insufficient unless the profiler follows the actual
GPU worker from its first runtime initialization and produces a worker receipt.
A parent-only trace can be parseable while omitting the executing process. The
harness must reject that result rather than infer inheritance from a command
line.

### Rejected: use `rocprofv3 -- <command>`

The retained `rocprofv3` script has SHA256
`195ff5e6faf48a3abbc6f4db9f69dd598fe71fa9ff695ba2556d65af636fdc48`.
Its `run` function appends the tool and SDK libraries to `LD_PRELOAD` at lines
1142-1145. Lines 1146-1155 then set `ROCP_TOOL_LIBRARIES` and
`LD_LIBRARY_PATH`. The earlier process-start log proves that this command can
reach the child worker, but it does not satisfy this design's no-`LD_PRELOAD`
rule. The implementation must not invoke `rocprofv3 -- <command>`.

### Rejected: preload or intercept runtime APIs

`LD_PRELOAD`, a HIP shim, or an API replacement can change library resolution,
capture, synchronization, or numerical execution. Such a result would not
measure the production route.

### Selected: pre-import process-tree bootstrap

The launch owner starts a fresh Python production subprocess with the exact
manifest command array. It adds one verified bootstrap directory to the front
of `PYTHONPATH`. That directory contains only the tracked and hashed
`sitecustomize.py` bootstrap. The owner rejects `-S`, `-I`, `-E`, a different
Python executable, an unbound existing `PYTHONPATH`, and any preexisting
`sitecustomize` module. The hook removes its directory from `sys.path` after
startup so it cannot resolve production imports.

CPython 3.12 runs `site.main()` automatically unless `-S` is present. The
retained `/usr/lib/python3.12/site.py` has SHA256
`619436355cbe91c3b29681ab486ee63f5f2e0a7d6274dfae747bdb94e60bca55`.
Its lines 571-588 import `sitecustomize`, and lines 611-643 do so before the
user command executes. The implementation binds the production interpreter
by path, version, size, SHA256, and build ID before launch. It binds `site.py`
by path, size, and SHA256.

The bootstrap imports only Python standard-library modules. Before profiler
activation, it rejects `torch` or `vllm` in `sys.modules`. It also rejects any
HIP, HSA, rocprofiler, or engine library in `/proc/self/maps`. The owner sets
the exact `ROCP_TOOL_LIBRARIES`, `ROCPROFILER_LIBRARY_CTOR`, output, category,
and local-prefix library values from the validated manifest. It sets no
`LD_PRELOAD` value. The bootstrap loads the exact extracted
`librocprofiler-sdk.so.1.1.0` with
`ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)`. It then calls
`rocprofiler_is_initialized` and requires a successful return with status `1`.
Only then does it write the root startup receipt and return control to CPython.
Because `site.py:571-588` reports and swallows ordinary hook exceptions, every
bootstrap failure writes a bounded failure receipt and calls `os._exit` with a
reserved nonzero status. User code must never run after a bootstrap failure.

The bootstrap registers an `os.register_at_fork(after_in_child=...)` callback.
The callback writes a child receipt before Python resumes the forked target.
It records the parent and child process IDs, start times, executable, process
group, profiler identities, initialization status, and loaded-library maps.
It rejects a child in which the SDK or tool mapping differs. The callback does
not replace the target, arguments, file descriptors, process group, result or
error channels, interprocess communication, signal handlers, or shutdown.

For production vLLM, manifest validation requires the pinned default `fork`
method. The unchanged supervisor imports `core.py`, constructs
`context.Process(target=EngineCoreProc.run_engine_core, kwargs=...)`, and calls
`proc.start()` as pinned earlier. The inherited active provider and the
at-fork receipt therefore exist before the worker resumes any imported runtime
code. A `spawn` or `forkserver` observation fails readiness. For vllm.cpp, the
production public Python adapter is the GPU owner, so its root receipt must
precede its first library load.

The retained process-start log has SHA256
`9140e80cd1934b700b7d982a9d33b931f1a04cd2fd8aa008f795a2e71616938d`.
Lines 87-110 show inherited tool registration. Lines 222-258 show profiler
initialization before the production command. Lines 688-1076 identify child
PID 55455 and its V2 initialization. Lines 1414 and 2016-2019 associate that
child with kernel records and its result file. Lines 2054-2117 show parent
finalization failure. This is observed reachability evidence only. Readiness
must reproduce the ordering with the selected no-`LD_PRELOAD` bootstrap and
must finalize cleanly.

The harness identifies the GPU owner from profiler records, the at-fork or
root receipt, and observed GPU activity. All three identities must agree. A
missing callback, changed start method, import before activation, parent-only
trace, or absent child provider receipt fails readiness. The implementation
does not fall back to attach, `LD_PRELOAD`, or a modified EngineCore target.

This process-tree form is used identically on both arms. Engine-specific code
may decode existing logs or emit diagnostic markers, but it cannot change
model inputs, scheduler decisions, tensor types, graph policy, sampling, or
worker lifetime.

## Public diagnostic contract

The implementation adds the importable
`tools.bench.strix_worker_profile.launch` seam. Its contract identifier is
`vllm.cpp/profiled-process-tree-launch/v1`. The seam accepts one validated
launch request with an exact production argument array, engine identity,
process-role binding, environment allowlist, profiler binding, lifecycle
limits, and output directory. It returns the same structured result that the
artifact schema defines later in this document.

The callable owns manifest validation, package and binary binding, bootstrap,
subprocess lifecycle, receipts, bounds, finalization, and artifact checks. It
validates and consumes one `vllm.cpp/runtime-loader-closure/v1` result before
bootstrap. It has no score-mode, benchmark verdict, or engine-specific
performance semantics. A caller must supply the complete production argument
array and the expected supervisor and GPU-owner roles. The callable rejects an
unknown schema version, a missing role, or an engine command that differs from
the bound manifest.

This seam owns the reusable worker-start launch requirement for downstream
profiling, including issue #3077. A downstream caller must require the exact
`v1` contract. It must fail closed on a version or result-schema mismatch. The
caller cannot copy the bootstrap or bypass the seam's validation and lifecycle
owner.

The implementation also adds one repository command:

```text
python3 tools/bench/strix_worker_profile/worker.py \
  --phase discover-worker-image|prepare-runtime-closure|readiness|trace \
  [--discovery-request <discovery-request.json>] \
  [--manifest <manifest.json>] \
  [--worker-image-receipt <worker-image-discovery-receipt.json>] \
  [--worker-image-receipt-sha256 \
    <worker-image-discovery-receipt.json.sha256>] \
  [--engine vllmcpp|vllm] \
  --output <new-directory>
```

The command is a thin client of the importable seam. It parses arguments,
constructs the launch request, invokes the callable once, and maps its result
to the process exit status. It does not implement separate validation,
bootstrap, lifecycle, closure, or artifact logic. The preparation phase invokes
the public runtime-closure callable exactly once. The readiness and trace phases
invoke the public launch seam exactly once.

The discovery and preparation phases forbid `--engine`. The readiness and
trace phases require it. Discovery requires `--discovery-request` and
rejects `--manifest`. Preparation requires the manifest and worker-image
receipt plus its detached SHA256 file. Every phase rejects arguments that its
selected phase does not own.

The discovery command invokes
`vllm.cpp/worker-image-discovery/v1` exactly once. The preparation command
invokes `vllm.cpp/runtime-loader-closure/v1` exactly once after the receipt
and detached-hash gate passes. A missing, malformed, mismatched, or swapped
pair must prevent that call. The command accepts no implicit engine, model,
profiler, or workload defaults.
Arguments stored in the manifest are arrays and are never interpolated through
a shell. The output directory must not exist. Hardware phases require
`RC_DEVICE=strix:gpu0` and a nonempty `RC_JOB_ID`. The command rejects symlinks,
paths outside its declared worker-local and NAS roots, inherited tuning
variables, `LD_PRELOAD`, eager or V1 switches, and an unbounded profiler
command.

The manifest schema identifier is
`vllm.cpp/strix-qwen3-worker-profile/v1`. It contains:

- the engine name, complete source revision, source-archive path and SHA256,
  build recipe, executable path, library paths, and expected identities;
- the profiler harness revision and source archive, nested SDK source revision
  and version, executable, libraries, build IDs, configuration bytes and
  SHA256, resolved categories, and output limits;
- the model repository, revision, dtype, file paths, byte counts, and SHA256
  values;
- the six prompt byte strings and their hashes, tokenizer identity, sampling
  fields, concurrency schedule, warmup schedule, and expected 128-token stop;
- the permitted environment, ROCm library roots, device, lease, timeout,
  per-file limit, aggregate-output limit, and cleanup timeout;
- the `vllm.cpp/worker-image-discovery/v1` contract and a
  `worker_image_baseline` object with its exact six-member schema, campaign ID,
  candidate receipt SHA256, qualification-manifest SHA256, envelope SHA256,
  and exact 12 binding fingerprints;
- the runtime-closure schema, closed SONAME binding map, exact selected source
  files, provenance kinds, pinned worker image, closure limits, and required
  source categories;
- the current worker-image discovery receipt path, byte count, SHA256, lease,
  boot, candidate receipt SHA256, and context relation;
- the closure manifest, archive, and receipt paths, byte counts, and SHA256 values;
- the sealed interpreter and host-glibc file identities; and
- the `vllm.cpp/profiled-process-tree-launch/v1` seam version, exact production
  command arrays, bootstrap file identity, multiprocessing method, and expected
  supervisor and worker lifecycle roles.

Unknown fields, duplicate JSON keys, missing full revisions, relative paths,
and schema-version drift fail before a subprocess starts. The implementation
records the raw manifest and its SHA256 in every phase result.

Both engines must use the same worker-image receipt and the same closure
manifest and archive hashes. The pair validator rejects different discovery
schemas, receipt hashes, closure schemas, limits, worker-image identities,
host-glibc identities, or fixed-point file and edge sets.

## Lifecycle and observation window

One owner controls this sequence:

1. Verify the lease, worker-image receipt, NAS free space, manifest, archives,
   runtime-closure receipt, model files, executable, libraries, profiler,
   build IDs, environment, and new output directory.
2. Record a pre-run binding manifest, the boot ID, device identity, lease job,
   process limits, and monotonic and wall-clock start times.
3. Extract the sealed closure into a fresh root. Recompute its fixed point,
   require `UNBOUND_COUNT=0`, and byte-match the host glibc set.
4. Start the exact production argument array through the pre-import bootstrap.
   Require the root receipt before any engine import. For vLLM, also require
   the at-fork child receipt before `EngineCoreProc.run_engine_core` resumes.
5. Run the declared warmup. Open the recorded observation window only after
   warmup and close it after the last matched request completes.
6. Run the complete ordered corpus. Emit correlated phase, request-dispatch,
   request-complete, and scheduler-step markers. Record actual request, token,
   padded-batch, graph-batch, and active-sequence shapes from the executing
   scheduler or runner.
7. Request profiler finalization while the owner still controls the process
   tree. Wait within the cleanup bound for each required worker artifact to
   close and become parseable.
8. Record the worker's finalization receipt. Then allow normal production
   shutdown and record the worker and supervisor exit statuses.
9. Recheck every bound file and build ID. Hash each artifact, write a failure
   result for any discrepancy, and publish the complete result atomically only
   after every required check succeeds.

Profiling may observe process startup and graph capture, because those records
are required to prove readiness. Kernel-time summaries and request attribution
use only the bounded corpus window. Warmup activity is labeled separately and
never folded into the corpus. An early end-of-sequence, missing request,
duplicated marker, reordered request, or requested-versus-actual shape mismatch
fails the arm.

## Readiness and trace phases

`readiness` uses a fresh production process and the same profiler tool,
configuration, model, and route as `trace`. It runs one declared warmup and one
prompt at concurrency one. It has a 10-minute wall timeout, a 256 MiB aggregate
output stop threshold, and a 192 MiB per-file limit. It proves only:

- the fixed campaign commitment comes from the reviewed acceptance head and
  matches the accepted manifest and candidate lineage;
- the worker-image receipt matches the current lease and boot;
- profiler initialization precedes runtime initialization;
- the sealed closure replays to the identical fixed point with
  `UNBOUND_COUNT=0` before bootstrap;
- the interpreter and host glibc set match the sealed identities;
- the selected launch seam, bootstrap hash, exact argument array, process
  roles, and `fork` method match the manifest;
- the profiler follows the actual GPU worker;
- every required semantic category produces a parseable record;
- graph capture or the resolved no-graph decision is explicit;
- a request marker joins to actual scheduler shapes and GPU activity; and
- finalization precedes normal shutdown.

Both arms must pass readiness before either full trace starts. A failed
readiness run is preserved and stops the campaign. It is not retried blindly.

`trace` uses a new production process for each arm. It runs the same declared
warmup and complete six-prompt, 128-token corpus at concurrency one and four.
Each arm has a 30-minute wall timeout, a 1 GiB aggregate-output stop threshold,
a 768 MiB per-file limit, and a 60-second cleanup timeout. The whole paired
stage has a 75-minute lease budget. These are stop thresholds, not exact disk
quotas; the monitor can observe a bounded overshoot between samples. A larger
bound requires a new reviewed design and a NAS capacity check.

Run the two trace arms sequentially inside one `rc` lease, on the same device,
with no unrelated GPU job. Alternate the first arm on a later reproduction if
the pair must be repeated. Never use SSH in place of the lease.

## Artifact schema

Each phase writes bounded logs and a final `result.json`. A failure writes
`result.failed.json` with the same provenance fields and the known partial
artifacts. It never renames that file to a passing result.

`result.json` contains:

- schema identifier, phase, engine, run identifier, manifest path and SHA256,
  start and end times, boot ID, lease job, device, and status;
- worker-image discovery schema, receipt path and SHA256, runtime-relevant
  image digest, operating-system identity, dpkg identity, and the exact 12 live
  records;
- fixed campaign-commitment path, canonical digest, acceptance state,
  candidate receipt SHA256, accepted profile-manifest SHA256, and exact 12
  `WorkerFileBindingFingerprintV1` values;
- launch-seam identifier, bootstrap path and SHA256, Python and `site.py`
  identities, exact production argument array, and resolved start method;
- runtime-closure schema, manifest, archive, and receipt identities, file and
  edge counts, total bytes, replay audit, and `UNBOUND_COUNT`;
- pre-run and post-run bindings for sources, model files, executable, every
  loaded engine and profiler library, configuration, and build IDs;
- the process tree with process IDs, start times, executable identities,
  lifecycle roles, exit statuses, and the uniquely identified GPU owner;
- profiler-start, runtime-start, worker-ready, observation-window,
  finalization, worker-exit, and supervisor-exit receipts in causal order;
- the resolved production mode, V2 status for vLLM, graph policy, capture
  sizes, replay mode, dtype, sampling values, and inherited environment;
- every warmup and corpus request, token counts, marker IDs, requested
  concurrency, actual scheduler shapes, graph selection, and matching trace
  correlation IDs;
- counts for HIP API, kernel, HSA/AQL, graph-capture, graph-replay, and marker
  records, including dropped-record and parser-diagnostic counts;
- a list of every artifact with relative path, semantic type, byte count,
  SHA256, producing process, finalization state, and parse result; and
- a fail-closed checklist whose fields name the evidence that satisfied each
  requirement.

Artifact paths are relative to the new output directory. The harness refuses
special files, symlinks, path escapes, duplicate paths, zero-byte required
files, artifacts still open at finalization, unrecognized process owners, and
any file whose recorded length or hash changes during the post-run check.

## Completeness rules

A trace arm passes only when all of these claims are supported by records from
the identified GPU worker:

- the fixed tracked campaign commitment came from the independently reviewed
  acceptance head and matches the accepted profile manifest, candidate receipt,
  envelope digest, and exact 12 binding fingerprints;
- the worker-image receipt matches the current lease, boot, image, operating
  system, dpkg database, and exact 12 live records;
- profiler initialization happened before Torch, HSA, or HIP initialization;
- the closure manifest and archive match the profile manifest, replay reaches
  the identical fixed point, and `UNBOUND_COUNT=0`;
- each closure mapping comes from the six-package root or sealed non-glibc
  root, except for the byte-matched host glibc set;
- no loader mapping uses an unsealed live-library fallback;
- the launch seam and bootstrap match their pinned versions, `LD_PRELOAD` is
  absent, and the exact production argument array ran without replacement;
- production vLLM used `fork`, the pinned `EngineCoreProc.run_engine_core`
  target, and matching root and at-fork child receipts;
- the production engine mode, graph defaults, resolved graph mode, dtype,
  sampling, model, binaries, libraries, and build IDs match the manifest before
  and after the run;
- all six prompts ran once in their declared order at both concurrency levels,
  with 128 generated tokens unless the manifest's declared stop reason applies
  identically to both engines;
- every request and scheduler step has a unique external correlation marker
  and recorded actual shape;
- HIP API, kernel dispatch, HSA/AQL, and marker streams are parseable and have
  no dropped or truncated tail records;
- graph capture events recorded before the corpus remain visible, and every
  corpus replay joins to its dispatched kernels and request marker;
- a resolved no-replay route is explicit evidence, not the absence of graph
  records;
- worker trace files finalized before normal worker and supervisor shutdown;
  and
- both arms use byte-identical profiler configuration and workload schedules.

A trace that starts after graph capture, contains only parent-process events,
misses a category, lacks actual scheduler shapes, omits pre-existing graph
identity, leaves an unfinalized temporary file, exceeds a bound, or requires a
parser warning to ignore corruption fails. Partial records remain diagnostic
and may guide the next design, but cannot support attribution.

## Diagnostic timing and accepted timing

Profiler timestamps may describe ordering and trace shares within one arm.
They do not become the throughput denominator and must not be compared as an
accepted performance ratio. Instrumentation overhead can differ across process
trees and kernels.

Only after both complete traces identify one measured, reachable difference may
a separate implementation spec select a lever. Any performance acceptance then
requires uninstrumented, same-binary A/B runs on an idle leased host, the
declared correctness gate, identical model and workload, warmed repetitions,
and reported concurrency-one and concurrency-four throughput, latency, and
memory. This design changes no threshold or tolerance and declares no ceiling.

## CPU test-first implementation

The fresh implementer first adds
`tests/tools/test_strix_worker_profile.py`. The initial focused run must fail
because the public entry point or required guard does not exist. Preserve that
red result. Implement only enough to make the focused suite pass, then run the
full gate.

CPU tests use temporary files and real bounded subprocesses where lifecycle
ordering matters. They simulate the profiler and production process tree; they
do not claim GPU coverage. At minimum they prove:

- bootstrap rejects every replay-only field, and replay rejects a missing or
  malformed accepted manifest, candidate receipt, or detached candidate hash;
- bootstrap produces only `SEALED_CANDIDATE`, and preparation rejects that
  candidate before package extraction even when its detached hash is valid;
- replay and preparation return `PENDING_CAMPAIGN_BASELINE` for an absent,
  dirty, malformed, or not-reviewed tracked commitment before any current-host
  observer runs;
- the production request and command reject a caller-supplied commitment path,
  bytes, digest, or override. A test-only fixture loader remains private and
  cannot be selected through the public surface;
- replay rejects a caller-swapped candidate receipt, profile manifest, or
  lookalike commitment. A fake current host that supplies its own baseline
  still fails before observation when the tracked commitment is absent;
- both request types reject a caller-supplied current lease, device, boot,
  image, package, or library identity;
- discovery rejects a missing controller lease identity, wrong device, invalid
  kernel boot-ID bytes, or another boot-identity source;
- relation fixtures cover same job and boot, fresh job and same boot, and fresh
  job and changed boot. Every unexpected job or boot relation fails;
- replay compares a newly observed envelope digest and all 12 binding
  fingerprints with the loaded campaign commitment. A current observation can
  never replace an expected value;
- worker-image discovery seals exactly 12 live records and rejects a missing
  path, multiple candidate, missing dpkg owner, changed image, changed
  operating-system or dpkg bytes, or changed path, hash, build ID, package, or
  symlink for any one record;
- one table-driven envelope suite enumerates all nine top-level members and
  every member of `ByteFileV1`, the package-row object,
  `QualificationBindingV1`, `WorkerFileBindingV1`, `ByteFileRefV1`, and the
  symlink-chain object. For each object member, it removes the member, changes
  one type-valid value, and injects a duplicate JSON key. For each array, it
  removes, changes, duplicates, and reorders one element. Every case must fail
  parsing or change the canonical bytes and digest, then fail baseline replay;
- one shared table enumerates all 19 `WorkerFileBindingV1` members plus every
  nested symlink and dpkg-list member. Removing or changing any member, changing
  primitive normalization, reordering a preserved array, duplicating a key or
  identity, changing the terminal line feed, or omitting the fingerprint domain
  changes the fingerprint and fails both receipt and commitment comparison;
- receipt fixtures enforce the exact 15-member receipt, exact three-member
  comparison object, null bootstrap lineage, replay lineage, and shared ordered
  12-fingerprint inventory. Commitment fixtures enforce its exact ten-member
  schema and the same fingerprint inventory;
- envelope fixtures reject an alternate domain, schema, key encoding, string
  normalization, integer encoding, base64 encoding, array order, duplicate
  identity, missing terminal line feed, or non-lowercase digest;
- authority fixtures prove that changing a lease job or boot according to the
  declared relation leaves the envelope digest unchanged. Injecting authority
  fields into the envelope fails the exact-member parser;
- deterministic discovery fixtures reproduce byte-identical receipt and hash
  bytes. Failures at each file fsync, directory fsync, rename, and parent fsync
  never create a usable final receipt;
- closure preparation rejects a missing receipt, missing detached hash,
  receipt-hash mismatch, bootstrap candidate, wrong lease or boot, changed
  image, campaign-lineage mismatch, or a change to any one of the 12 bindings
  before package extraction;
- two byte-identical complete final receipt directories receive the same
  preparation result even when a fixture constructs one without the discovery
  publisher. This proves that preparation does not claim retrospective
  knowledge of atomic publication;
- the public preparation command requires `--worker-image-receipt-sha256` and
  delegates its exact 65-byte encoding and pair checks to the request decoder.
  Missing, mismatched, malformed, and crossed receipt and hash paths prevent
  `prepare()` from running;
- a valid receipt and hash pair reaches the public preparation callable once,
  with the exact receipt bytes and decoded 32-byte hash in
  `RuntimeClosureRequest`;
- closure preparation cannot read the loader cache, enumerate a live library
  directory, run dpkg ownership discovery, select another path, or publish a
  replacement worker-image receipt;
- strict manifest parsing rejects duplicate keys, unknown fields, incomplete
  pins, wrong hashes, path escapes, symlinks, existing outputs, and engine or
  profiler mismatches;
- package recovery rejects a missing package, wrong byte count, wrong package
  field, wrong payload hash, wrong ELF build ID, unresolved dependency, or
  dependency outside the allowed roots;
- the eight-unbound reviewer fixture is RED before closure preparation;
- a sealed synthetic closure reaches `UNBOUND_COUNT=0` at acquisition and replay;
- closure preparation rejects an omitted transitive dependency, duplicate
  SONAME target, changed symlink, hash, build ID, archive member, or package
  identity;
- closure preparation rejects an undeclared SONAME, a binding to the wrong
  source class, ambiguity within the selected class, a selected-path change,
  or substitution between DEB-payload, qualification-manifest, and live-dpkg
  provenance;
- closure preparation rejects host glibc mismatch, either resource limit,
  closure growth, live fallback, or a nonzero unbound count;
- the second walk must reproduce the selected canonical files and ordered
  dependency edges, not only the file count;
- deterministic fixtures reproduce byte-identical manifest, archive, and
  receipt outputs after source mtimes, owners, and enumeration order change;
- the pair validator rejects different profiler configuration bytes,
  categories, tool libraries, workload bytes, resource bounds, or closure
  hashes;
- the environment guard rejects eager, V1, `LD_PRELOAD`, tuning variables, and
  unexpected runtime library roots;
- a fake supervisor and GPU worker preserve their arguments, file descriptors,
  exit status, and shutdown order under the launch owner;
- the public command reaches the importable
  `vllm.cpp/profiled-process-tree-launch/v1` callable exactly once, and schema
  drift fails before any subprocess starts;
- the preparation command reaches the importable
  `vllm.cpp/runtime-loader-closure/v1` callable exactly once;
- the discovery command reaches the importable
  `vllm.cpp/worker-image-discovery/v1` callable exactly once;
- production call-site reachability fails if the launch seam stops consuming
  the sealed closure result;
- a real CPU fixture starts a fresh Python interpreter with the hashed
  `sitecustomize` bootstrap, records activation before user code, and uses an
  at-fork receipt before the unchanged child target executes;
- the CPU fixture proves the exact argument array, target, keyword arguments,
  inherited file descriptors, process group, result and error pipes, exit
  status, and shutdown order are unchanged;
- `-S`, `-I`, `-E`, `spawn`, `forkserver`, preloaded engine libraries, a
  swallowed bootstrap exception, and any `LD_PRELOAD` value fail before user
  code;
- a startup receipt after a simulated runtime-init marker fails;
- a parent trace without the GPU-worker receipt and GPU activity fails;
- missing HIP API, kernel, HSA/AQL, graph-capture, graph-replay, marker, actual
  shape, or pre-existing graph identity fails independently;
- a legitimate explicit no-replay decision remains distinguishable from a
  missing graph stream;
- duplicated, reordered, early-stopped, or incomplete corpus requests fail;
- dropped records, an unclosed file, a zero-byte required file, a parse error,
  a changing post-run hash, or finalization after worker exit fails;
- timeout, aggregate-size, per-file, fatal GPU diagnostic, and cleanup failures
  preserve bounded evidence and fail the command; and
- readiness cannot be reused as a full trace or as a timing result.

The fresh reviewer mutates each guard and its production call site in a scratch
copy. At minimum, remove the pre-runtime ordering check, worker ownership check,
package-set validation, closure fixed-point check, zero-unbound check, no-live-
fallback check, callable-to-command connection, closure-to-launch connection,
at-fork receipt, one category check, graph-replay join, scheduler-shape join,
finalization-order check, post-run binding check, output bound, and eager or
preload refusal one at a time. Remove the discovery receipt or its detached
hash. Let a bootstrap candidate reach preparation. Remove or bypass the fixed
campaign-commitment loader. Accept a missing or not-reviewed commitment, a
caller-supplied override, a caller-swapped carrier, or a fake current host that
supplies its own baseline. Accept a changed image or changed one of the exact
12 bindings. Replace a replay expectation with the current observation. Change
each context relation and replace the kernel boot source. Omit, alter,
duplicate, and reorder every envelope member through the shared parameterized
inventory. For every `WorkerFileBindingV1` member, remove it from the
fingerprint preimage or change its normalization. Remove the fingerprint domain
or terminal line feed. Give the receipt and commitment different fingerprint
inventories. Remove the prepare hash flag, bypass the callable-owned decoder,
accept a malformed detached file, and swap two valid receipt and hash pairs.
Make preparation rediscover one binding. Make preparation distinguish two
byte-identical final directories by their unobservable publication history.
Each mutation must fail independently. Mutate an omitted transitive dependency,
symlink, file hash, build ID, archive hash, host glibc mismatch, resource limit,
live fallback, source class, selected path, provenance kind, undeclared SONAME,
and second-walk edge independently. The focused suite must detect each
mutation. Restore the tree byte-for-byte after every mutation.

## Hardware stages and gates

No hardware stage belongs to this design-only change. The later implementation
uses these gates in order:

1. Focused CPU red:
   `python3 -m unittest tests.tools.test_strix_worker_profile -v`.
2. Focused CPU green with the same command.
3. Full repository preflight with `scripts/agent-preflight.sh`.
4. Fresh static and mutation review of the immutable implementation commit.
5. Operator rerun of the focused suite and full preflight.
6. One leased bootstrap `discover-worker-image` phase. It initializes no GPU
   runtime and atomically publishes an unaccepted `SEALED_CANDIDATE`. No later
   phase can consume that candidate directly.
7. One operator promotion change at
   `.agents/campaign-baselines/BACKEND-GATE-ROCM-SGLANG-3076.json`. It records
   the exact candidate and accepted-manifest identities. The real replay and
   hardware gate remains `PENDING` until the commitment is committed and a
   fresh reviewer passes its immutable head.
8. One leased replay `discover-worker-image` phase. It loads the fixed tracked
   commitment before host observation. It validates the caller-carried
   candidate receipt and accepted manifest, observes the declared context
   relation, and atomically publishes the `SEALED` replay receipt.
9. One `prepare-runtime-closure` phase in the same lease and boot. It loads the
   commitment itself and consumes only the replay receipt for the image and 12
   live records. It atomically publishes the bounded closure.
10. Offline validation and extraction of the sealed closure. The replay audit
   must reach the identical fixed point with `UNBOUND_COUNT=0`.
11. One leased readiness arm for vllm.cpp, then one for production vLLM.
12. One leased full trace per arm, sequentially, using the same accepted
   manifest and profiler configuration.
13. An offline pair check that rehashes both outputs, verifies every completeness
   rule, and emits a diagnostic comparison without a performance verdict.

Stages 6 and 8 through 12 use one lease and boot when operator promotion and
review finish within the held lease. Otherwise, stage 8 starts a fresh lease
and declares `fresh-lease-same-boot` or `fresh-lease-changed-boot` as observed.
Before any later stage starts in a fresh lease or after a reboot, rerun stages
8 through 10. The tracked commitment supplies baseline authority. The
caller-carried candidate receipt supplies bytes for validation only.

The hardware report records every command exit, omitted gate, resource stop,
and failed completeness rule. A passing CPU suite cannot replace a hardware
receipt. An implementer or reviewer report cannot replace the operator's gate.

## Risks

- Profiler output can exceed the bounded budget before a sampled monitor reacts.
  Per-file and aggregate limits bound the overrun and preserve failure evidence.
- Production vLLM can spawn more than one Python child. Process-tree identity,
  startup receipts, and GPU records must agree on one executing worker.
- Profiler finalization can deadlock or fail after useful records exist. The
  cleanup timeout preserves partial artifacts but does not convert them to a
  passing trace.
- Diagnostic markers can perturb timing. They are accepted only for correlation,
  and instrumented timing is never the acceptance denominator.
- A reboot can invalidate worker-local builds. The recovery path rebuilds from
  the verified package set and sealed closure. It rebinds every binary and
  library before bootstrap.
- A fresh lease or reboot invalidates the old receipt for preparation. The old
  bootstrap receipt remains candidate evidence named by the tracked campaign
  commitment. The new discovery must match that commitment's image digest and
  all 12 records before preparation can run.
- A copied glibc can corrupt a running Python process. The launch seam never
  loads the archived glibc witnesses and requires byte-identical host files.
- A live system fallback can make one run pass and the rebooted run fail. The
  replay audit rejects every non-glibc mapping outside the sealed roots.
- Current NAS and local free space are narrow. The implementation must check
  capacity before each phase and must not duplicate model artifacts.

## Stop conditions

Stop without attribution or optimization when any of these occurs:

- the Strix lease is absent, lost, or shared with an unrelated GPU job;
- discovery cannot derive the complete image envelope, or its lease, device,
  boot, or runtime-relevant digest differs;
- replay or preparation cannot load the fixed tracked campaign commitment from
  the independently reviewed acceptance head;
- the commitment is missing, dirty, not reviewed, malformed, or inconsistent
  with its candidate receipt, accepted manifest, envelope, or 12 fingerprints;
- replay lacks the candidate receipt and detached hash, uses another boot
  source, or observes a lease and boot relation that its request does not
  declare;
- replay replaces a baseline digest or binding with a current observation, or
  a second bootstrap attempts to continue the accepted campaign;
- worker-image discovery does not seal exactly 12 records or cannot publish
  its receipt and detached hash with one atomic directory rename;
- preparation lacks the sealed worker-image receipt or detached hash, accepts
  a malformed, mismatched, or swapped pair, accepts a self-authorized current
  host, accepts a bootstrap candidate, accepts a caller-selected commitment,
  or tries to rediscover a live binding;
- the sealed image, operating-system bytes, dpkg identity, or any one of the
  12 sealed path, hash, build-ID, package, or symlink records changes;
- a pin, archive, model file, binary, library, build ID, configuration, or
  before-and-after binding differs;
- the package set is incomplete, its dependency closure is unresolved, or a
  dependency lacks the required hash and build-ID receipt;
- closure preparation does not reach a stable fixed point with
  `UNBOUND_COUNT=0` within 64 ELF files and 512 MiB;
- the closure manifest, archive, receipt, package ownership, symlink chain, or
  deterministic reproduction differs;
- the closed SONAME map is incomplete, selects another source class or path, is
  ambiguous within its selected class, or changes a selected edge on replay;
- the target interpreter or host libc, libm, dynamic loader, libdl, libpthread,
  or librt differs from the sealed identity;
- a non-six-package dependency resolves outside the sealed closure, or a live
  fallback supplies it;
- the launch seam or result schema differs from
  `vllm.cpp/profiled-process-tree-launch/v1`;
- the pinned profiler cannot start before the production GPU worker initializes
  its runtime or cannot follow that worker without attach or interposition;
- `sitecustomize` does not run first, the vLLM start method is not `fork`, or
  the at-fork child receipt does not precede the unchanged worker target;
- the production V2 or normal vllm.cpp route cannot run with the declared
  defaults;
- a graph mode, actual scheduler shape, request marker, trace category, or
  worker owner is ambiguous;
- any record is dropped, corrupt, truncated, unfinalized, or exceeds a bound;
- the full corpus or token contract is incomplete or differs across arms;
- a GPU fault, host reboot, cleanup failure, or normal-shutdown failure occurs;
  or
- an observed difference requires a product decision outside this spec.

Preserve the bounded failure record and name the exact missing authority or
mechanism. Do not switch to attach, eager mode, V1, another profiler, a larger
unreviewed resource bound, or a different engine pin. Complete matched traces
are the prerequisite for the next decision, not evidence that any particular
kernel should change.
