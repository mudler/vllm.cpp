# Downloadable server binary release matrix

Status: accepted contract with required W1-W11/W13 implementation complete for
`ENG-RELEASE-BINARIES` in the single delivery PR #196. The release row remains
`ACTIVE`: v0.0.2 published all eight primary tuples, while matching-hardware
evidence and the native Windows v0.0.3-pre.1 extension remain pending. W12 is
optional/non-primary.

Pins: vLLM parity source `555967922`; vllm.cpp spike baseline `f13c49ee`;
request [#117](https://github.com/mudler/vllm.cpp/issues/117); hosted CI repair
[#447](https://github.com/mudler/vllm.cpp/issues/447); claim
`CLAIM-ENG-RELEASE-BINARIES-SPIKE` in draft PR
[#129](https://github.com/mudler/vllm.cpp/pull/129); W5 implementation claim
`CLAIM-ENG-RELEASE-BINARIES-W5` in draft PR
[#141](https://github.com/mudler/vllm.cpp/pull/141); W6 implementation claim
`CLAIM-ENG-RELEASE-BINARIES-W6` in draft PR
[#196](https://github.com/mudler/vllm.cpp/pull/196). The Windows extension's
per-commit documentation checkpoint repair is tracked by
[#448](https://github.com/mudler/vllm.cpp/issues/448); its archive-target
checkpoint repair is tracked by
[#450](https://github.com/mudler/vllm.cpp/issues/450). The merged-SHA dry-run
repairs for exact prerelease identity and the native MSVC test translation unit
are tracked by [#499](https://github.com/mudler/vllm.cpp/issues/499) and
[#500](https://github.com/mudler/vllm.cpp/issues/500), specified together in
[release-dry-run-gate-repairs.md](release-dry-run-gate-repairs.md).

## Delivery topology

Developer direction on 2026-08-09 binds all remaining W1-W13 work to PR #196.
Each work unit remains an independently committed, red-first, executable
checkpoint, but no work unit is split into another implementation PR. PR #196
stays draft until every required work unit, the full release dry run, fresh
review, and operator verification are green. W12 remains optional and cannot
block W13; its optional status does not permit moving it to another PR.

<!-- release-binary-contract:begin -->
identity=ENG-RELEASE-BINARIES
lifecycle=ACTIVE
manifest_schema=vllm.cpp.release-manifest.v1
delivery_pull_request=196
delivery_mode=single-pr-W1-W13
primary_cuda_artifact=one-fat-binary-per-os-host-abi
primary_cuda_sms=80,86,87,89,90a,100a,103a,110,120a,121a
per_sm_cuda=optional-non-primary
primary_cpu_artifact=one-adaptive-binary-per-os-host-abi
x86_64_baseline=portable-sse2-without-avx2
work_W12_policy=optional-non-blocking
archive_claims=published-v0.0.2
published_tag=v0.0.2
published_sha=7020de93652ca920424a10ac5255b34810dd2f24
published_run=31466516224
published_asset_count=26
runtime_claims=pending
metal_channel=stable-after-runtime-gate
mlx_channel=preview
vulkan_channel=preview
musl_channel=experimental-preview
musl_scope=cpu-only-no-gpu
rocm_channel=blocked
gpu_driver_boundary=external-host-never-bundled
required_anchor_paths=.agents/engine-matrix.md,.agents/roadmap_v1.md,.agents/NOW.md,.agents/coordination.md,.agents/completed/state-events/2026-08/STATE-20260809T160000-001.md,release/manifest-v1.schema.json,scripts/release_manifest.py,tests/scripts/test_release_manifest.py,examples/CMakeLists.txt,scripts/package-server.py,tests/scripts/test_server_package.py
work_W1=
work_W2=W1
work_W3=
work_W4=
work_W5=
work_W5_status=implemented
work_W6=
work_W6_status=implemented
work_W7=W1,W2,W3,W4,W5,W6
work_W8=W5,W7
work_W9=W3,W4,W5,W6,W7
work_W10=W1,W2,W5,W6,W7
work_W11=W5,W6,W7
work_W12=W1,W2,W5,W6,W7
work_W13=W5,W7,W8,W9,W10,W11
<!-- release-binary-contract:end -->

The block above is consumed by `scripts/check-release-binary-contract.py`; its
values and the human-readable work table below must change together through a
new reviewed design decision. In particular W12 is optional and is deliberately
absent from W13's dependency set.

The same checker owns its execution path: it requires itself and its 30-test
mutation suite as direct commands in an unconditional CI step, and executes an
instrumented copy of preflight to prove both named arrays reach their real
loops. The suite independently mutates the Metal, MLX, Vulkan, musl, ROCm and
external-driver policies above; quoted, conditional, inert or deleted wiring is
not accepted as execution.

## Scope and product contract

The deliverable is a downloadable, backend-specific `vllm-server` bundle. The
server links the vllm.cpp core statically, while unavoidable platform runtimes
remain explicit dependencies. A release bundle is an installed staging tree,
not a copy from a build directory. It contains at least:

- `bin/vllm-server`, linked to the static `vllm` core;
- `VERSION`, a machine-readable release manifest, SHA256 checksum, SPDX JSON
  SBOM, build-provenance attestation, and third-party license notices;
- runtime files that the selected lane is licensed and designed to redistribute
  (for example MLX's dylib/metallib in the MLX preview lane); and
- no model weights, tokenizer assets, Python, PyTorch, Triton runtime, compiler,
  or source/build directory.

The canonical primary archive name is
`vllm.cpp-<version>-<artifact-id>.tar.gz`. `<version>` is exactly the semantic
version in the release plan and embedded manifest; `<artifact-id>` is copied
verbatim from `release/release-matrix.json`, whose stable tuple IDs encode the
OS, host architecture/ABI, and backend. For example, v0.0.2's x86 glibc CPU
archive is `vllm.cpp-0.0.2-linux-x86_64-glibc-cpu.tar.gz`. Its checksum and
provenance names are derived by appending `.sha256` and `.provenance.json` to
that complete archive name. Package targets, lane drivers, handoff inventory,
release-index generation/verification, workflow paths, documentation, and
publication all use this one spelling; reordered or unversioned aliases are not
release assets. SHA-bound GitHub Actions artifact names are transport container
identities and do not replace or alter filenames inside them.

One binary never crosses an OS or host ABI. The primary CPU download is
one conservative-baseline, runtime-adaptive binary per OS+host ABI; the primary
CUDA download is one fat binary per OS+host ABI containing every supported SM. Its
manifest makes compiled CPU tiers or CUDA SMs explicit. Optional per-SM CUDA
archives are diagnostic/performance variants, not the primary downloads. A
future server install component and package target must stage this exact tree.
The current CMake installs
`libvllm.{a,so}` and `vllm.h` but does not install the server
(`CMakeLists.txt:1712-1783`; `examples/CMakeLists.txt:54-64`), so the install and
package targets are implementation work, not present-tense capability.

Out of scope for this spike: implementing CMake or CI, publishing a release,
bundling models, promising cross-libc portability, and widening any backend,
model, quantization, or kernel support claim.

## Upstream chain

Pinned vLLM's structural release reference is
`.buildkite/release-pipeline.yaml:1-18,34-170`; its CPU release-image dependency
boundary is `docker/Dockerfile.cpu:262-290`. Those lanes establish that release
construction, validation, and publication are separate concerns. vllm.cpp does
not copy Python wheels or containers: it applies the same separation to native
installed server archives while vLLM remains the runtime-behavior oracle.

## Our baseline

The current server is the CMake target `server`, linked to the static `vllm`
target and gated only by a help smoke (`examples/CMakeLists.txt:54-64`). W5 now
provides the versioned schema and deterministic build-time generator/validator
(`release/manifest-v1.schema.json`, `scripts/release_manifest.py`) with
synthetic CPU/CUDA fixtures; those fixtures are contract tests, not artifact or
runtime evidence. The
library install rules package `libvllm.a`, the shared C ABI library, and
`include/vllm.h` (`CMakeLists.txt:1712-1783`). There is no server install rule,
archive layout, staged dependency audit, provenance, or publish workflow.
Existing CUDA AOT trees are exactly the six directories named
below. Cross-family CUDA fat builds currently fail because feature sources are
gencode'd for the whole list, and `cmake/TritonAOT.cmake:94-126` currently
rejects multi-arch AOT; both are prerequisites, not accepted release limits.

The CPU baseline already demonstrates the intended shape. CMake applies ISA
flags surgically to selected translation units rather than globally
(`CMakeLists.txt:870-890` and the immediately following per-source blocks).
Element GEMM chooses portable/NEON or SSE2+F16C/AVX2/AVX-512 tables at runtime
and exposes a forced-tier test seam
(`src/vt/cpu/cpu_matmul_elem.cpp:553-612`). Arm quant i8mm is separately
compiled and selected through Linux HWCAP2 or Darwin sysctl
(`src/vt/cpu/cpu_quant_dot_arm.cpp:39-77`). This is a partial foundation, not
proof that every CPU kernel has every tier. ROCm's opt-in skeleton has never
compiled on AMD hardware.

## Evidence classes and publication policy

The channel is evidence-driven, not backend-driven:

| Channel | Required evidence | User-facing promise |
|---|---|---|
| `stable` | clean build, staged-archive validation, matching-hardware runtime smoke, representative correctness gate, and release-tag rerun all pass for the exact target tuple | supported downloadable bundle for the named tuple |
| `preview` | clean build and staged-archive validation pass; runtime, correctness, or performance evidence is absent or incomplete | build-only/testing-welcome artifact; no runtime-support or performance claim |
| `blocked` | the backend does not yet compile or lacks the minimum packaging/runtime contract | no artifact is published |

Build success never sets runtime evidence true. Runtime smoke never sets
correctness or performance evidence true. The release manifest carries
independent booleans and evidence URLs for `build_verified`,
`archive_smoke_verified`, `dependency_audit_verified`, `runtime_verified`,
`correctness_verified`, and `performance_verified`. It also records the exact
commit, clean-tree status, compiler/toolchain, CMake cache options, target
architecture or CPU baseline/tier set, host ABI, dependency versions, and test
commands. CUDA evidence is per SM and CPU evidence is per compiled tier. Stable
publication fails closed unless every stable-required boolean is true;
preview publication preserves false values rather than deriving or omitting
them.

Performance evidence is informational in an archive manifest until the normal
same-box, same-workload vLLM/competitor gate has run. This release row cannot
turn a build result into a throughput claim.

## Release matrix

The initial matrix is deliberately hybrid. Runtime-gated tuples may graduate
to stable; build-only tuples remain downloadable previews so users can test
hardware the project does not own.

| Artifact tuple | Initial channel | Backend flags and evidence boundary |
|---|---|---|
| `linux-x86_64-glibc-cpu` | stable after baseline and tiered runtime gates | one adaptive binary: SSE2/portable baseline without AVX2, plus only inventoried per-TU F16C/AVX2/AVX-512 and later VNNI/AMX tiers whose kernels and exact probes exist; record glibc/libstdc++ floors and compiled tiers |
| `linux-aarch64-glibc-cpu` | stable after baseline and tiered runtime gates | one adaptive binary: NEON/portable baseline plus only inventoried HWCAP/HWCAP2-gated DotProd/i8mm and future tiers whose kernels exist; independent arm64 evidence is never inferred from x86_64 |
| `linux-x86_64-glibc-cuda` | preview until the fat prerequisite, archive gates, and per-SM evidence land | primary x86_64 CUDA download; explicit SM set `80,86,87,89,90a,100a,103a,110,120a,121a`; per-source gencode and per-SM runtime/AOT dispatch required |
| `linux-aarch64-glibc-cuda` | preview until the fat prerequisite, archive gates, and per-SM evidence land | primary aarch64 CUDA download with the same ten SM device targets but a distinct host ELF ABI; cannot be the x86_64 archive |
| `macos-arm64-metal` | stable after M-series runtime gate | native Metal, MLX off; record deployment target and required system frameworks |
| `macos-arm64-metal-mlx` | preview until its exact bundled MLX tuple is runtime/correctness-gated | Metal plus opt-in MLX provider and redistribution/license audit; record MLX dylib and metallib versions |
| `linux-x86_64-glibc-vulkan` | preview | Vulkan explicitly on; loader/device/driver remain external; only the Vulkan-supported model/quant surface is declared |
| `linux-x86_64-musl-cpu-static` | experimental preview | literal-static feasibility lane; CPU only; see the static boundary below |
| ROCm/HIP | blocked | `VLLM_CPP_HIP` skeleton has never compiled on AMD hardware; no archive until compile, staged smoke, and matching-hardware gates exist |

`AUTO` is forbidden in release presets. Every backend/provider is selected
explicitly from the canonical CMake surface: `VLLM_CPP_CUDA`,
`VLLM_CPP_CUDA_ARCHITECTURES`, `VLLM_CPP_METAL`, `VLLM_CPP_MLX`, `MLX_ROOT`,
`VLLM_CPP_VULKAN`, `VLLM_CPP_HIP`, `VLLM_CPP_HIP_ARCHITECTURES`,
`VLLM_CPP_SERVER`, `VLLM_CPP_BUILD_EXAMPLES`, `VLLM_CPP_BUILD_TESTS`, and
`VLLM_CPP_TRITON` (`CMakeLists.txt:24-120`; `cmake/TritonAOT.cmake:52-87`).
The manifest records resolved feature-table output and compiled CPU tiers; it
does not say “all acceleration,” “all CUDA features,” or “all models.” No
release preset uses global `-march=native`.

### Primary CUDA artifacts: fat per OS and host ABI

The KISS primary download is one fat CUDA binary for each OS+host ABI, not one
download per SM. The first two are separate Linux x86_64-glibc and Linux
aarch64-glibc archives. Both explicitly build the complete supported set
`80;86;87;89;90a;100a;103a;110;120a;121a`; their device code may match, but
their host ELF ABI cannot. A future OS/libc tuple gets another artifact rather
than widening either binary into an impossible universal host executable.

Three current gaps must close before either primary CUDA bundle can build:

1. **Per-source gencode.** The current cross-family fat build applies the whole
   target list to incompatible fast-path translation units, so sm12x PTX is
   rejected for other families. CMake must compile portable sources for every
   supported target and narrow each feature source to the exact SM intersection
   declared by `VT_CUDA_FEATURE_TABLE`. The fat-build gate inspects every TU's
   SASS/cubin set; a green final link alone is insufficient.
2. **Multi-SM Triton AOT packaging.** `cmake/TritonAOT.cmake:94-126` currently
   rejects multiple architectures because it selects one vendored tree. The
   release prerequisite extends the builder/embedding contract to include every
   available per-SM tree in one host binary/archive, namespace its launchers,
   and select the exact cubin from the runtime device capability. It must never
   load a same-family “close enough” cubin.
3. **Runtime dispatch.** Device capability selects only a compiled compatible
   tactic and exact AOT tree. Missing optimized code takes the documented
   portable C++/CUDA fallback; an unsupported device fails loudly. The manifest
   lists compiled SASS, fast-path feature cells, AOT availability, and separate
   build/runtime/correctness/performance evidence for every SM.

The honest AOT/evidence inventory remains:

| CUDA SM | AOT embedded in fat artifact | Current evidence and optional per-SM variant |
|---|---:|---|
| `80` | yes | preview; derived/build-verified, no matching runtime board |
| `86` | yes | preview; same-major derived/build evidence, no matching runtime board |
| `87` | no | preview; portable synchronous path runtime-verified, default async path remains a known bug |
| `89` | yes | preview; derived/build-verified, no matching runtime board |
| `90a` | yes | preview; derived/build-verified, no matching runtime board |
| `100a` | yes | preview; derived/build-verified, no matching runtime board |
| `103a` | no | preview; portable build-only target |
| `110` | no | preview until the archive repeats the existing portable runtime proof |
| `120a` | no | preview; build-supported, no matching runtime board |
| `121a` | yes | stable candidate after extracted-archive smoke, both gate-model correctness, and release performance rerun on GB10 |

The six complete vendored AOT trees are exactly `80`, `86`, `89`, `90a`,
`100a`, and `121a`. For `87`, `103a`, `110`, and `120a`, the fat binary records
AOT unavailable and uses the portable CUDA fallback. No lane fabricates or
borrows another SM's cubin. Optional single-SM archives may still be published
for diagnosis, size comparison, or peak-performance experiments, using the
same evidence labels; they are not the primary downloads and cannot substitute
for either host-ABI fat-build gate.

The overall fat artifact stays preview while any included path has only
build evidence. Per-SM manifest evidence remains independent, so the existing
sm_121a stable candidate and build-only targets are not flattened into a
blanket support claim.

### Primary CPU artifacts: one adaptive binary per OS and host ABI

CPU packaging follows the same KISS rule at the host-ABI level: one binary per
OS+host ABI, compiled to a conservative baseline and dispatching at runtime.
There is no artifact per ISA tier, no global `-march=native`, and no binary that
crosses Linux/macOS, glibc/musl, x86_64/aarch64, or deployment-target ABIs.

For x86_64, the baseline must run without AVX2: portable/SSE2 code remains
callable, and higher instructions live only in per-function or per-TU tiers.
The current element-GEMM selector already exposes SSE2+F16C, AVX2, and AVX-512F
tiers (`src/vt/cpu/cpu_matmul_elem.cpp:553-612`), while CMake's per-source ISA
shape avoids changing the rest of the binary's baseline
(`CMakeLists.txt:870-890`). The release audit inventories the exact instruction
requirements of every tier. AVX/AVX2/AVX-512 require both CPU feature bits and
OS-enabled extended state; VNNI, AMX, BF16, or other tiers are added only where
a real kernel exists, with every required CPUID bit and OS state (including
XCR0/tile permission where applicable). The manifest never turns “CPU has
AVX-512” into “all kernels are AVX-512 accelerated.”

For aarch64, the baseline is portable/NEON and optimized quant tiers are chosen
only after exact OS capability probes. Linux uses `getauxval(AT_HWCAP*)`;
Darwin uses the corresponding `sysctl` keys. DotProd and i8mm each require their
own compiled kernel and probe; the existing i8mm path demonstrates the required
compile/runtime split (`src/vt/cpu/cpu_quant_dot_arm.cpp:39-77`). SVE/SVE2,
SME, BF16, or later tiers remain absent from the manifest until kernels, exact
HWCAP/sysctl probes, and gates exist.

Every adaptive CPU artifact records its conservative baseline, compiled tiers,
per-tier kernel families, required CPU bits, required OS state, detected tier,
and selected tier. Forced-tier mutation tests must execute each compiled path on
feature-rich hosts and force the baseline on the same binary; on feature-poor
hosts or emulation they must prove the baseline executes and forcing an
unsupported tier fails closed before an illegal instruction. Feature detection
alone is not execution evidence.

## Static and external-runtime boundary

The normal bundles are **static-core**, not “one literal static executable.”
`vllm-server` contains the static project core but may dynamically depend on the
host C/C++ runtime, pthreads, platform frameworks, or a selected accelerator
runtime. Those dependencies must be enumerated and audited from the staged
binary.

The one literal-static experiment is
`linux-x86_64-musl-cpu-static`. It is CPU-only and passes only when `file`
identifies a static executable, `ldd` reports no dynamic interpreter, the server
help and loopback health smoke pass in a minimal container, and DNS/thread/file
loading behavior is exercised. It stays experimental preview even when green;
the result decides whether a second arm64 musl lane is warranted. It follows the
same conservative x86_64 adaptive-tier contract but remains a distinct musl ABI,
not a replacement for the glibc artifact. The archive must not silently disable
server functionality to obtain a static link.

Accelerator drivers are honest external boundaries. NVIDIA's kernel driver and
CUDA driver ABI, the Vulkan loader/ICD and device driver, macOS Metal system
frameworks, and the ROCm kernel/user runtime cannot be made portable by
statically linking the vllm.cpp core. The archive manifest names the minimum
tested driver/runtime; it never claims to bundle a GPU driver. MLX is an
opt-in preview exception whose redistributable dylib/metallib may be carried
only with its exact license and version.

`ffmpeg` is also external. The server's video path defaults to the `ffmpeg`
executable and spawns it from the example boundary
(`examples/server/main.cpp:215,349-350,741-758,839,1090-1093`). Bundles do not
silently vendor it. The manifest and README say that text serving needs no
ffmpeg and video generation requires a compatible executable on `PATH` or an
explicit `--video-ffmpeg` path. Models, tokenizer data, certificates, and GPU
drivers are runtime inputs, not archive payloads.

## Gates: staged archive and supply chain

Every CI lane builds, installs into an empty staging prefix, creates the archive,
extracts it into a second empty directory, and validates only that extracted
tree. A build-tree smoke is not release evidence.

Required gates:

1. **Package contents:** exact allowlist, executable bit, no absolute build
   paths, no source/object files, no credentials, and version output matching
   the tag, commit, manifest, and C ABI version.
2. **Server smoke:** `vllm-server --help`; bind loopback on an ephemeral port;
   `/health`, `/version`, and clean shutdown; then a small representative model
   request on lanes with matching runtime hardware.
3. **Dependency audit:** Linux `readelf` plus `ldd`/`lddtree`, macOS `otool`, and
   platform equivalents reject an undeclared shared object, build-directory
   RPATH/RUNPATH, absolute developer path, or missing library. ELF RPATH must be
   absent or relative to the extracted bundle; Mach-O install names must use
   system paths, `@rpath`, or `@loader_path` as declared.
4. **CUDA fat-architecture audit:** `file`/ELF headers match the host ABI;
   `cuobjdump` proves all ten named SM targets are present in the primary fat
   binary; per-TU inspection proves each fast-path source contains only its
   compatible gencode set. All six available AOT trees are embedded and resolve
   only on their exact SM; the other four capabilities select the portable
   fallback. Removing any one SM/AOT mapping or routing one to a neighbour is a
   red mutation.
5. **CPU adaptive-dispatch audit:** dependency/manifest data lists baseline,
   compiled tiers, exact CPU bits and OS-state prerequisites per kernel family.
   Forced-tier mutations cover every tier. The extracted binary executes on
   feature-poor and feature-rich hosts or faithful emulation: baseline works
   without AVX2 on x86_64, NEON/portable works without optional Arm extensions,
   rich tiers execute when supported, and unsupported forced tiers refuse before
   an illegal instruction. Build commands reject global `-march=native`.
6. **Correctness:** CPU unit/conformance tests before packaging; after
   extraction, representative endpoint and model checks. Each CUDA SM and CPU
   tier retains independent evidence. The sm_121a path owes both project gate
   models and the normal oracle comparison before its stable-candidate flag can
   turn true; another SM's result cannot satisfy it.
7. **Supply chain:** archive SHA256, SPDX JSON SBOM, source/dependency/license
   inventory, third-party notices, immutable build provenance, and a `VERSION`
   record containing tag, commit, clean-tree bit, compiler, backend, target,
   host ABI, and C ABI version. The checksum and provenance refer to the final
   archive bytes, not the staging directory.

The release manifest schema and its checker are versioned together. Missing
evidence is `false` with a reason; command failure cannot collapse into “not
applicable.”

## Port map

| Responsibility | Local destination | Contract |
|---|---|---|
| server install/package component | top-level and `examples/CMakeLists.txt` | stage the existing static-core server under its canonical output name without changing server behavior |
| release tuple/preset | focused release CMake presets or matrix data | explicit backend/provider/host ABI and exhaustive ten-SM fat list; no `AUTO`, implicit wildcard, `-march=native`, or blanket feature claim |
| CUDA fat compile/dispatch | CMake feature-source gencode plus CUDA/AOT runtime selector | narrow every fast-path TU to compatible SMs; embed all available AOT trees; dispatch exact SM with portable fallback where AOT is unavailable |
| CPU adaptive compile/dispatch | per-source ISA options and CPU selectors | conservative ABI baseline plus only real kernel tiers, guarded by exact CPU and OS-state probes |
| manifest and supply-chain metadata | release scripts plus a versioned schema | independent evidence values, final-archive SHA256, SPDX SBOM, provenance, version and licenses |
| staged archive validator | release checker tests/scripts | validate extracted bytes, dependencies, RPATH/install names, host architecture and AOT SM |
| dry-run/tag automation | release workflow | build, verify, attest and publish as isolated least-privilege stages |
| release index | generated public release notes/index | derive channel and limitations from verified manifests, never handwritten assumptions |

## Tests to port

The pinned upstream release pipeline is the executable structural spec; its
separate build/test/publish stages are mirrored by W8 rather than porting a
Python runtime test. Existing local executable specs remain binding:

- `test_server_help` (`examples/CMakeLists.txt:59-63`) runs from the extracted
  archive, not only the build tree;
- endpoint conformance under `tests/vllm/entrypoints/openai/` supplies the
  server protocol smoke before a lane can become stable;
- each backend row's existing correctness suite and matching-hardware model
  gate supplies runtime evidence; cross-builds record runtime false; and
- new manifest/archive fixtures mutate every required evidence field, RPATH,
  dependency allowlist, architecture and AOT association to prove the checker
  fails for the named reason;
- CUDA mutations remove each SM in turn, mis-assign an AOT tree, or widen a
  fast-path TU's gencode and must fail before publication; and
- CPU mutations force every compiled tier, remove each feature/OS-state probe,
  and run the extracted binary on feature-poor and feature-rich hosts/emulation.
  Unsupported force requests must fail closed; supported force requests must
  execute the named tier, not merely report it.

## Least-privilege CI and release flow

The same build definition serves pull-request dry runs and tags, but authority
is separated by stage:

1. **Plan:** read-only checkout computes the matrix and validates the tag/version
   relationship. A pull request or manual dry run cannot create a release.
2. **Build:** per-tuple jobs have `contents: read`, no token write scope, no OIDC,
   and upload temporary workflow artifacts only. Cross builds produce build
   evidence, never runtime evidence.
3. **Verify:** fresh jobs download and extract archives, run the staged checks,
   and emit the independent evidence manifest. Hardware jobs receive only the
   exact tuple assigned to that runner.
4. **Attest:** only the provenance job receives `id-token: write`; it signs the
   verified archive digest, not arbitrary workspace content.
5. **Publish:** a protected tag/environment job alone receives
   `contents: write`. It downloads verified immutable archives, checks their
   digests and evidence channel, and attaches them to the matching release.

Fork pull requests never receive release secrets. Tag names are untrusted input
until the version gate passes. Artifact names are allowlisted, publish uses no
wildcards, and a failed lane cannot be replaced by an older workflow artifact.

## Dependencies

- the existing server and library install boundaries;
- canonical CMake backend flags, CUDA feature-table resolution, and per-source
  gencode narrowing for the full ten-SM list;
- multi-tree Triton AOT embedding/namespacing plus exact runtime selection of
  the six complete per-SM trees;
- CPU per-source/per-function ISA compilation, exact OS capability APIs, and
  feature-poor plus feature-rich hosts or emulation for both host architectures;
- matching runtime hosts for any stable tuple and the normal GPU contention
  protocol for correctness/performance gates;
- redistribution-compatible licenses for every bundled runtime file;
- platform dependency-inspection tools plus an SPDX SBOM/provenance generator;
  and
- protected release environments for attestation and publication authority.

## Work breakdown: helper-sized implementation plan

Each work unit is a separate verified checkpoint inside the single active
claim and PR #196, with its own red-first checker change and review. W5 was
implemented in PR #141; required W1-W11/W13 are implemented in #196. Hosted
completion remains necessary before any publication or channel claim.

| Work | Deps | Deliverable | Exit gate |
|---|---|---|---|
| W1 | — | cross-family CUDA fat-build prerequisite: per-source gencode narrowing over all ten supported SMs | clean x86_64-host fat build; per-TU `cuobjdump` mutation proves every compatible SM present and every incompatible SM absent |
| W2 | W1 | multi-SM Triton AOT embedding, namespacing, manifest and exact runtime dispatch | all six available trees coexist in one fat binary; exact-SM dispatch tests plus portable fallback for the four unavailable trees; wrong-tree mutation red |
| W3 | — | x86_64 CPU ISA-dispatch inventory and completion | SSE2/portable baseline runs without AVX2; current F16C/AVX2/AVX-512 tiers forced and executed; exact OS-state probes; VNNI/AMX listed only for real gated kernels; no `-march=native` |
| W4 | — | aarch64 CPU ISA-dispatch inventory and completion | NEON/portable baseline plus independently forced DotProd/i8mm where kernels exist; exact Linux HWCAP/Darwin sysctl gates; poor/rich host or emulation execution |
| W5 | — | **IMPLEMENTED (#141):** versioned release-manifest generator and schema with independent per-SM and per-CPU-tier evidence | 19/19 tests: fixtures distinguish absent, false, failed and true; schema type/enum/const are independently live; booleans cannot satisfy integer types/constants; CPU tier kernel-family/bit/OS-probe inventories are exact; compiled tiers/SMs, fallback/AOT state, the named NVIDIA driver dependency, publication channels, static boundaries and host ABI are mandatory |
| W6 | — | **IMPLEMENTED (#196):** canonical `vllm-server` output name, install component, and deterministic staging/package target for the existing static-core server | clean CPU build installs into an empty prefix; archive bytes reproduce; extracted `--help` runs without dynamic `libvllm`; existing library/header install remains present |
| W7 | W1, W2, W3, W4, W5, W6 | staged archive validator: allowlist, dependency/RPATH, fat-SM/AOT and adaptive-CPU audits, SHA256, VERSION, licenses and SPDX SBOM | Linux fixture/archive tests red-first; no build paths, missing SM, unsafe ISA tier or undeclared dependency accepted |
| W8 | W5, W7 | least-privilege dry-run/tag workflow, immutable artifact handoff, provenance and protected publish stages | permissions checker plus dry run proves no release is created and publish cannot consume unverified bytes |
| W9 | W3, W4, W5, W6, W7 | primary adaptive CPU bundles: Linux glibc x86_64+aarch64 and experimental x86_64 musl literal-static | glibc binaries execute baseline and supported rich tiers on matching hosts/emulation before stable; musl remains preview |
| W10 | W1, W2, W5, W6, W7 | primary Linux CUDA fat bundles for x86_64 and aarch64 host ABIs | each extracted archive contains all ten SMs and six exact AOT trees; per-SM evidence remains independent; no host ABI is inferred from the other |
| W11 | W5, W6, W7 | macOS arm64 native-Metal/MLX and Linux Vulkan bundles | native Metal runtime-gated; MLX/Vulkan preview until exact archive gates; dependencies and install names audited |
| W12 | W1, W2, W5, W6, W7 | optional single-SM CUDA diagnostic/performance variants | generated from the same explicit matrix and evidence; never advertised as the primary KISS download or used to bypass W10 |
| W13 | W5, W7, W8, W9, W10, W11 | release index/docs and retention policy generated from manifests | every link, checksum, channel, host ABI, compiled tier/SM, driver boundary and limitation matches published bytes |

ROCm remains blocked outside these work units until its backend row first
compiles on AMD hardware. A new lane is added by changing this matrix and its
tests before workflow expansion, never by a wildcard build.

## Risks and decisions

- “Static” without the `static-core` qualifier is rejected for normal GPU and
  platform bundles; only the musl CPU experiment may say literal-static.
- Stable is a property of an exact artifact tuple and evidence set, not of a
  backend family. A later toolchain or dependency change reruns the gates.
- Cross-compilation can prove bytes and architecture, not execution. Preview is
  the honest publication channel for community hardware coverage.
- A fat CUDA link is not enough: per-source cubin inventory and exact runtime
  dispatch are gates. Optional single-SM builds cannot substitute for them.
- CUDA fast-path availability differs by SM. The feature-table output and AOT
  state are manifest data; archive names do not imply feature parity.
- CPU feature detection is not enough: each compiled tier must execute under a
  forced-tier test, and every instruction requires its CPU plus OS-state gate.
  One adaptive binary still cannot cross an OS or host ABI.
- ffmpeg and accelerator drivers are external operational dependencies. Their
  absence must fail the relevant feature actionably, not corrupt a general
  server smoke.
- Reproducibility means a recorded clean recipe plus immutable provenance first;
  byte-for-byte rebuild reproducibility is a separate evidence field until
  demonstrated.

## Hosted CUDA archive-validation repair

The 2026-08-10 hosted dry run at Actions run `31363264184` built and packaged
six of eight tuples successfully. Both CUDA tuples completed their ten-SM
builds and archive creation, then failed the extracted-archive dependency gate:
`ldd` resolved `libcuda.so.1` through `build-release-cuda-{x86,arm64}`. The
publish chain correctly stopped before verify, attest, or publish.

The failure is in the validation harness, not in the archive. CUDA's driver is
an external host dependency and hosted build runners have no driver runtime, so
the harness creates a controlled `libcuda.so.1` alias for the extracted
`--help` smoke. The caller currently creates that alias below the same build
directory passed to `--forbid-path`, making the harness violate its own
invariant. The validator must continue rejecting every dependency resolved
through a source or build tree.

The approved repair is deliberately narrow:

1. The Linux accelerator release driver creates the validation-only CUDA stub
   directory with `mktemp` outside the build tree and removes it on exit.
2. `prepare-cuda-driver-stub.sh` continues to resolve the CUDA toolkit's
   external stub and create only the runtime SONAME alias; it does not copy or
   bundle that stub into the release archive.
3. `validate-release-archive.py` and its strict forbidden-path check remain
   unchanged. No build path is allowlisted and no `ldd` result is suppressed.
4. A red-first regression test proves the release driver no longer places the
   validation stub below `$build_dir`, still exports the exact controlled
   runtime directory, and installs cleanup before validation.
5. The focused release archive/accelerator/workflow tests, repository
   preflight, fresh static+mutation review, and a new hosted eight-tuple dry run
   must pass. Only that hosted run may advance `archive_claims`; tagged
   publication remains a separate developer-authorized action.

Rejected alternatives are allowing the known stub path through the validator
or replacing the runtime resolution smoke with `readelf` alone. The former
weakens the no-build-path release invariant, while the latter stops proving
that the extracted executable's declared dependencies resolve.

## Hosted artifact-handoff completion

The 2026-08-10 manual dry run at Actions run `31408404388` proved that all
eight required platform jobs build, validate, package, and upload their exact
bundle triplets. The aggregate `build` job `93565669335` then failed before
handoff verification with
`[Errno 2] No such file or directory: 'plan/release-plan.json'`.
`actions/download-artifact@v4` had extracted the exact plan artifact below an
additional artifact-name directory because that download did not set
`merge-multiple: true`; the consumer intentionally reads the stable flat path
`plan/release-plan.json`.

The same latent layout mismatch applies to every later exact single-artifact
handoff. `verify` reads both `plan/release-plan.json` and
`unverified/release-handoff.json`; `attest` and `publish` read the verified
handoff and assets from fixed paths. The asset-set download already opts into
flat extraction and succeeded. Fixing only the first observed failure would
therefore defer the same failure to verify or to the first real tag run.

The approved completion is one invariant across the whole workflow:

1. Every `actions/download-artifact@v4` step uses `merge-multiple: true`,
   including exact single-artifact downloads. Artifact names stay immutable and
   SHA-bound, and consumer paths stay stable and explicit.
2. `scripts/check-release-workflow.py` fails unless the flattening invariant is
   present on every download step. It continues requiring exact artifact names,
   explicit paths, immutable handoffs, least-privilege permissions, and
   wildcard-free publication.
3. `tests/scripts/test_release_pipeline.py` first demonstrates a red mutation
   by removing or falsifying one download's flattening flag, then proves the
   repaired workflow green. Existing workflow and release-pipeline mutations
   remain green.
4. The focused workflow checker and mutation suite, full preflight, fresh
   static plus scratch-mutation review, and operator rerun must pass before the
   branch is pushed. A new manual dry run must then reach the `verify` job with
   all eight tuples and an immutable verified handoff.
5. Manual `workflow_dispatch` remains non-publishing by design: successful
   build and verify may advance archive evidence, but cannot prove OIDC
   attestation or GitHub Release publication. Those stages require a real
   signed/authorized `v*` tag whose version matches `CMakeLists.txt`, followed
   by an audit that every matrix archive, checksum, SBOM/provenance sidecar, and
   generated release index was attached from the verified handoff.

Rejected alternatives are duplicating artifact-name subdirectories throughout
consumer paths or adding discovery/move scripts. Both repeat generated names
outside their producing expressions and weaken the fixed-path handoff contract.
Universal flat extraction is the action's native mechanism and keeps the
workflow's exact-file publication boundary unchanged.

## Spike verdict

The release program is feasible as backend-specific static-core bundles with a
hybrid stable/preview channel. Literal-static scope is limited to the
experimental musl CPU lane. ROCm is blocked. Primary downloads are adaptive CPU
or fat CUDA per OS+host ABI; per-SM CUDA archives are optional diagnostics. The
Required W1-W11/W13 are implemented and the row is `ACTIVE`, not `DONE`. Local
archive, CPU, Vulkan, workflow, and mutation gates are green. The v0.0.2 tag
workflow published the eight archive/checksum/provenance triplets and two
generated indexes (26 assets) from
`7020de93652ca920424a10ac5255b34810dd2f24` in run `31466516224`.
Matching-hardware evidence and the Windows v0.0.3-pre.1 extension remain
pending. W12 remains the optional non-primary diagnostic lane.

## Outcome

PR #446's hosted CPU gate exposed two release-version call sites still pinned
to `0.0.2` after the project advanced to `0.0.3`, plus the
`vllm-server-archive` CMake target omitting the packager's required explicit
archive format. Issue #447 updates both executable expectations to `0.0.3` and
passes `--archive-format tar.gz` at the existing deterministic archive target;
the archive remains a tarball and no release workflow behavior changes.

That first local #447 repair changed the public CMake archive target without an
atomic `docs/USAGE.md` projection, so the per-commit documentation checkpoint
correctly rejected it ([#450](https://github.com/mudler/vllm.cpp/issues/450)).
The replacement preserves the version and explicit-format fix and documents
the developer archive separately from prerelease workflow asset naming; no
checker or workflow behavior is weakened.

## Now

**ACTIVE; required W1-W11/W13 implemented and v0.0.2 published.** The eight
primary archives and their sidecars/indexes are live from SHA
`7020de93652ca920424a10ac5255b34810dd2f24` (run `31466516224`, 26 assets).
Matching-hardware evidence and the Windows v0.0.3-pre.1 extension remain
pending.
