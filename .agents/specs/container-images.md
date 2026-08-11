# Published container images on GHCR

Status: accepted design and IMPLEMENTED for W1-W5 and W7. The `cpu` lane is
built and gated end to end on a real image; `vulkan` and `cuda` are implemented
and gated statically but have not been built on this box. **No image has been
pushed: the GHCR package does not exist, and no lane has runtime evidence on
matching accelerator hardware (W6).**

Pins: vLLM parity source `555967922`; vllm.cpp baseline
`24306364ab8beaed9197604a042a56aaccfde493`; issue
[#170](https://github.com/mudler/vllm.cpp/issues/170); roadmap row `IMG`
(`ROAD-V1-CONTAINERS`); engine-matrix row `ENG-RELEASE-CONTAINERS`.

This spec is subordinate to the accepted
[release binary matrix](release-binary-matrix.md). The image **is** that
bundle: where the two disagree, the release contract wins and this document is
wrong.

## Scope and product contract

The deliverable is a pullable OCI image that serves an OpenAI-compatible API
with no setup beyond a model mount. One GHCR package,
`ghcr.io/mudler/vllm.cpp`, carries every lane; the lane is in the tag, never in
the package name.

| Lane | Immutable tag | Moving tag | Initial channel |
|---|---|---|---|
| `cuda` | `:<version>-cuda` | `:latest-cuda` | preview |
| `vulkan` | `:<version>-vulkan` | `:latest-vulkan` | preview |
| `cpu` | `:<version>-cpu` | `:latest-cpu`, `:latest` | stable after the baseline-tier gate |
| `rocm` | — | — | blocked; no image is published |

Every lane is a `linux/amd64` + `linux/arm64` manifest list whose members are
built on native runners. aarch64 is first-class rather than an afterthought
because the project's own gate hardware — GB10 (`sm_121a`), Thor (`sm_110`),
Orin (`sm_87`) — is arm64.

The image contains the staged release bundle, the lane's runtime libraries, and
`ffmpeg`. It contains no model weights, no tokenizer assets, no Python, no
PyTorch, no compiler, no source tree, and no build tree.

`:latest` aliases the `cpu` lane. A user who pulls the bare tag on a machine
with no accelerator gets a working server rather than a library-load failure.

### Boundaries, recorded once

- **The GPU driver is never bundled.** `libcuda.so.1` and the Vulkan
  ICD/loader-visible driver come from the host through the container runtime
  (`nvidia-container-toolkit` or an equivalent device plugin). The image
  declares the minimum tested driver; it never ships one.
- **Metal and MLX are NOT-CONTAINERIZABLE.** There is no macOS container
  runtime and no Metal device passthrough. `macos-arm64-metal` and
  `macos-arm64-metal-mlx` stay static-binary-only lanes permanently. This is a
  recorded boundary, not pending work, and no future work unit may quietly
  convert it into one.
- **ROCm is blocked**, tracking its binary channel. `BACKEND-ROCM` has not
  compiled on AMD hardware, so there is nothing to containerize. The lane is
  added by changing this spec and its matrix, never by a wildcard build.
- **Model weights are a runtime input**, mounted by the operator. No image
  variant bakes a checkpoint. Weights live on the operator's storage, per the
  standing checkpoint policy.

Out of scope: Kubernetes manifests, Helm charts, a Compose file, an
autoscaling story, a base image for third-party derivation, and any widening
of a backend, model, quantization, or performance claim.

## Upstream chain

The structural reference is pinned vLLM's release image lanes,
`.buildkite/release-pipeline.yaml:34-170`, and its published-image dependency
boundary, `docker/Dockerfile.cpu:262-290`, at `555967922`. Those establish that
image construction, validation, and publication are separate authorities, which
is the property mirrored here.

Nothing else is mirrored. vLLM's images exist to ship a Python wheel with
PyTorch and a CUDA runtime; ours ship one static-core native executable. vLLM
remains the runtime-behavior oracle and carries no authority over our image
layout.

## Our baseline

There is no server Dockerfile. The only container asset in the tree is
`docker/Dockerfile.arm64`, which builds `vllm-bench` and
`vllm-cpu-kernel-bench` for an arm64 CPU-kernel cross-check and exports them to
a `scratch` stage. It is not a release artifact, does not build the server, and
is untouched by this row.

The substrate this row consumes already exists and is green locally:

- `scripts/package-server.py` stages the canonical tree — `bin/vllm-server`,
  `VERSION`, `release-manifest.json`, `sbom.spdx.json`, `THIRD_PARTY_NOTICES`,
  and `share/licenses/**` — and rejects any undeclared file
  (`scripts/package-server.py:101-127`).
- `scripts/build-cpu-release.sh` and `scripts/build-linux-accelerator-release.sh`
  are the single build definitions per lane, including the ten-SM list
  `80;86;87;89;90a;100a;103a;110;120a;121a`
  (`scripts/build-linux-accelerator-release.sh:20-24`).
- Those scripts already **package and validate**: they call
  `release_accelerator_metadata.py`, `package-server.py --archive`, and
  `validate-release-archive.py` with a strict `--forbid-path` against the build
  directory, and for CUDA they prepare the driver stub outside the build tree
  first (`scripts/build-linux-accelerator-release.sh:66-113`).
- `.github/workflows/release.yml` already implements the least-privilege
  plan → build → handoff → verify → attest → publish chain over eight tuples,
  with `check-release-workflow.py` guarding it.
- The server defaults to `0.0.0.0:8000` (`src/vllm/entrypoints/openai/server_main.cpp:135-136`),
  the same default as vLLM, and exposes `/health` and `/version`
  (`src/vllm/entrypoints/openai/api_server.cpp:998,1002`).
- The video path spawns an external `ffmpeg`, overridable with
  `--video-ffmpeg` (`src/vllm/entrypoints/openai/server_main.cpp:330`).

The CUDA lane links `CUDA::cudart` and `CUDA::cublasLt`
(`CMakeLists.txt:1417`) plus `CUDA::cuda_driver` (`CMakeLists.txt:1894`).
Vulkan is `dlopen`'d, never linked (`CMakeLists.txt:1234`). That asymmetry
decides the runtime bases in the image topology.

## Decision: images are built, not repacked

Two designs were considered.

**Rejected — repack the published archive.** The image build downloads the
exact validated tarball the release workflow produced and unpacks it, so the
image is the archive byte-for-byte and inherits its evidence. Rejected because
it forces the container workflow to run strictly after, and to reach into, a
release run that has not yet published anything (`ENG-RELEASE-BINARIES` is
`ACTIVE` with a pending hosted dry run), and because the cross-workflow
artifact handoff is brittle: the image can only be cut when a release
publishes, and a re-cut needs the original run's artifacts to still exist.

**Accepted — build in the Dockerfile, calling the release scripts.** The
builder stage runs the same `scripts/build-*-release.sh` the release workflow
runs. There is one build definition, one staged layout, and no second place
where the ten-SM gencode list or the CPU tier flags are written down. Images
are buildable from any commit, locally and in CI, with no dependency on a
published release.

The cost is stated rather than glossed: for a given version the image bytes and
the published archive bytes are **separately built**. The image is the same
bundle *by construction*, not the same bytes. Attestation binds each
independently, and no document may claim the image digest and the archive
digest describe the same build.

A consequence in our favour: because the release scripts already end in
`validate-release-archive.py`, every image build runs the full archive audit —
allowlist, RPATH, dependency resolution, forbidden build paths, and for CUDA
the fat-SM inventory. That is not optional here and is not a separate work
unit. It also means the CUDA driver-stub handling repaired for hosted run
`31363264184` must work under BuildKit; that is a certainty to solve in W2, not
a risk to monitor.

## Image topology

One `docker/Dockerfile` with a shared builder pattern and per-lane targets, so
a lane cannot grow its own layout. Base images are pinned by digest, never by
floating tag.

```
FROM ubuntu:24.04@sha256:…             AS builder-cpu
FROM nvidia/cuda:<v>-devel-ubuntu24.04@sha256:…  AS builder-cuda
FROM ubuntu:24.04@sha256:…             AS builder-vulkan
    ARG VERSION SOURCE_SHA EVIDENCE_URL SOURCE_DATE_EPOCH JOBS
    RUN scripts/build-cpu-release.sh … | scripts/build-linux-accelerator-release.sh <id> <backend> <dir>
    # → build + package + validate; leaves the staged tree:
    #   bin/vllm-server VERSION release-manifest.json
    #   sbom.spdx.json THIRD_PARTY_NOTICES share/licenses/**

FROM ubuntu:24.04@sha256:…             AS cpu | vulkan | cuda
    COPY --from=builder /…/release/stage /opt/vllm
    # cuda:   + libcudart.so.12 + libcublasLt.so.12 (copied from the toolkit)
    # vulkan: + libvulkan1 (loader only; the ICD is the host's)
    # all:    + ffmpeg, ca-certificates
```

Builder requirements per lane, each an explicit W1/W2 obligation:

- `cpu` — `build-essential`, `cmake`, `ninja-build`, `python3`, plus the
  feature-poor emulator the CPU release script expects for its baseline-tier
  execution.
- `vulkan` — the same, plus a software ICD, because the script *runs*
  `test_vulkan_backend` and `test_backend_cross_device`
  (`scripts/build-linux-accelerator-release.sh:51-65`); a builder with no ICD
  fails the build rather than skipping the test.
- `cuda` — the CUDA toolkit at `/usr/local/cuda`, required both by `nvcc` and
  by `prepare-cuda-driver-stub.sh`. `VLLM_CPP_CUTLASS_FETCH=ON` means the build
  reaches the network; the fetched revision is pinned and recorded, or the
  build is not reproducible.

`-j 2` is hardcoded in the accelerator script
(`scripts/build-linux-accelerator-release.sh:55`). A ten-SM CUDA build at two
jobs will not finish inside a hosted runner's limits. W2 owes a `JOBS`
parameter threaded through both callers, changed in the script rather than
worked around in the Dockerfile.

### Runtime contract

```
ENTRYPOINT ["/opt/vllm/bin/vllm-server"]
EXPOSE 8000
USER 1000:1000
VOLUME ["/models", "/cache"]
HEALTHCHECK CMD  GET http://127.0.0.1:8000/health
LABEL org.opencontainers.image.{source,revision,version,licenses,description,created}
LABEL io.vllm-cpp.{lane,channel,cuda-sms,cpu-baseline,min-driver}
```

No `CMD` overrides host or port: the server already binds `0.0.0.0:8000`, so
flags after the image name pass straight to `vllm-server`. `/models` is the
weights mount, expected read-only; `/cache` is tokenizer/HF cache and must be
writable by uid 1000. The quick start is one line:

```sh
docker run --rm --gpus all -p 8000:8000 -v /models:/models:ro \
  ghcr.io/mudler/vllm.cpp:latest-cuda --model /models/qwen3
```

### ffmpeg: a deliberate divergence

The release-binary contract refuses to vendor `ffmpeg`
([release-binary-matrix.md](release-binary-matrix.md) § static and
external-runtime boundary). Images install it in **every** lane anyway, argued
here rather than waived silently: an archive is extracted onto a host that has
a `PATH` to inherit, and a container has none, so applying the archive rule to
images would ship a `/v1/videos` endpoint that can never succeed in the default
image.

The obligations that follow are not optional. `ffmpeg`'s licenses enter
`THIRD_PARTY_NOTICES` and the image SBOM; the distribution build's GPL/LGPL
components are named; and the smoke gate proves `ffmpeg` resolves on `PATH`
inside the image. `--video-ffmpeg` keeps working for operators who mount their
own.

## Gates and evidence

The channel model is inherited unchanged from the release contract: build
success never sets runtime evidence true, and runtime smoke never sets
correctness or performance evidence true. Evidence is per lane **and per
architecture** — an amd64 result never speaks for arm64.

1. **Build.** Each lane builds for both architectures on its native runner.
2. **Bundle audit.** Inherited from `validate-release-archive.py` inside the
   builder: allowlist, executable bit, no absolute build paths, no source or
   object files, no credentials, declared dependencies only, RPATH absent or
   bundle-relative, and for CUDA the ten-SM `cuobjdump` inventory.
3. **Image layout audit.** A new check over the final image: `/opt/vllm`
   matches the staged tree exactly, `VERSION` agrees with the tag and the C ABI
   version, no toolkit, compiler, source, or build directory survived into the
   runtime stage, and the image runs as uid 1000.
4. **Container smoke.** `--help`; run detached and `GET /health` and
   `/version`; `ffmpeg -version` resolves; SIGTERM produces a clean exit inside
   the grace period.
5. **Lane runtime evidence.**
   - `arm64-cuda` — a real model request on GB10 (`sm_121a`) through the
     project's GPU serialization protocol ⇒ runtime-verified for that tuple
     only.
   - `amd64-cuda` — no matching board; build evidence only ⇒ preview.
   - `vulkan` — a software-ICD smoke on the runner is build/preview evidence
     and is never reported as runtime support.
   - `cpu` — executes its conservative baseline under the same feature-poor
     emulator the release workflow installs, on both architectures.
   - Thor (`sm_110`) — see the risks section; until it runs, the arm64 cuda image makes no
     Tegra claim.
6. **Workflow mutation suite.** `scripts/check-container-workflow.py` with
   red-first tests, mirroring `check-release-workflow.py`. Deleting a smoke
   step, widening a job's permissions, pushing from a pull request, or
   re-pushing an existing immutable tag must each fail closed. Quoted,
   conditional, or inert wiring is not accepted as execution.
7. **`release/container-matrix.json`** (`vllm.cpp.container-matrix.v1`) — lanes
   × architectures × channels × pinned base-image digests × retention, with its
   own checker, mirroring `release/release-matrix.json`.

## Supply chain and least-privilege publish

`.github/workflows/containers.yml`, stage-separated exactly like `release.yml`:

1. **Plan** — read-only; resolves version, lane set, and base digests from the
   container matrix; validates the tag/version relationship. A pull request or
   dry run cannot publish.
2. **Build** — per lane × architecture, `contents: read`, no registry write, no
   OIDC. Pull requests stop after the smoke gates.
3. **Smoke** — the gate list above against the built image.
4. **Push by digest** — `packages: write`, this job only, on tags only. Each
   architecture pushes its own digest; no tag is written yet.
5. **Manifest** — `docker buildx imagetools create` joins the two architecture
   digests into `:<version>-<lane>`. A pre-push guard fails if that tag already
   exists, because GHCR does not enforce immutability for us.
6. **Attest** — `id-token: write`, this job only; `actions/attest-build-provenance`
   over the manifest digest, plus buildx `--provenance=mode=max`.
7. **Promote** — moves `:latest-<lane>`, and `:latest` for the cpu lane, only
   after every preceding stage is green for both architectures.

Triggers: release tags publish; pull requests build and smoke without pushing;
`workflow_dispatch` is a dry run. Nothing publishes between tags, so
`:latest-<lane>` always means the last release.

SBOM is two-part and both parts ship: the staged `sbom.spdx.json` describes the
bundle, and a buildx image SBOM describes the base image, `ffmpeg`, and the
copied CUDA runtime libraries. The bundle SBOM alone would misdescribe the
image.

Retention: immutable version tags are maintainer-deletion-only; `latest-*`
move; untagged digests are pruned on a schedule recorded in the container
matrix beside the existing `ci_artifacts_days: 7`. Fork pull requests never
receive registry credentials. Tag names are untrusted input until the version
gate passes, and no publish step uses a wildcard.

## Risks and open questions

- **Tegra versus SBSA.** A hosted `ubuntu-24.04-arm` runner produces an SBSA
  CUDA build. GB10 is SBSA and is expected to run it; **Thor (`sm_110`) and
  Orin (`sm_87`) are Tegra/L4T with a different CUDA runtime**. The arm64 cuda
  image is claimed to run on Tegra only once it has run on Thor. If it cannot,
  the honest outcome is a recorded boundary or a separate Tegra lane, never a
  quiet widening of the arm64 tag.
- **CUDA runtime redistribution.** Copying `libcudart.so.12` and
  `libcublasLt.so.12` out of the toolkit is permitted under the CUDA EULA's
  redistributable list, but the exact file list, version, and notice text are a
  W2 audit item with a named reviewer, not an assumption.
- **ffmpeg licensing.** The image topology states the divergence and its obligations; the
  GPL/LGPL component inventory of the distribution build is real work.
- **Image size.** The cuda lane is expected in the high hundreds of MB, driven
  by `libcublasLt` and the ten-SM fat binary. It is measured and recorded, not
  optimized speculatively.
- **BuildKit and the CUDA driver stub.** Certain, not hypothetical: the build-not-repack decision.
- **Trademark.** A public registry package named `vllm.cpp` is a more visible
  surface than a repository. Outreach to `collaboration@vllm.ai` has no reply.
  If the name must change, the change is a rename of one package and every tag
  in this spec; that is the recorded contingency.
- **CUTLASS fetch at build time.** Network access inside the image build makes
  the fetched revision part of the artifact. It is pinned and recorded in the
  manifest or the build is not reproducible.

## Port map

| Responsibility | Local destination | Contract |
|---|---|---|
| lane image definition | `docker/Dockerfile` | one file, per-lane targets, digest-pinned bases; a lane never gains its own build definition |
| lane build | existing `scripts/build-cpu-release.sh`, `scripts/build-linux-accelerator-release.sh` | called unchanged from the builder stage; a `JOBS` parameter is added in the script, never worked around in the Dockerfile |
| staged tree | existing `scripts/package-server.py` | the image's `/opt/vllm` is exactly the staged tree, allowlist included |
| bundle audit | existing `scripts/validate-release-archive.py` | already invoked by the build scripts; runs inside the image build, with the CUDA driver stub created outside the build tree |
| image layout and smoke audit | new release/container scripts plus tests | validate the final image, not the builder stage: staged tree, uid, labels, `ffmpeg` on `PATH`, `/health`, `/version`, clean SIGTERM |
| lane/arch/channel data | `release/container-matrix.json` plus its checker | lanes, architectures, channels, pinned base digests and retention are data, mirroring `release/release-matrix.json` |
| publish workflow | `.github/workflows/containers.yml` | seven least-privilege stages; tags publish, pull requests never do |
| workflow guard | `scripts/check-container-workflow.py` plus mutation tests | mirrors `check-release-workflow.py`; inert or conditional wiring is not execution |

## Tests to port

There is no upstream test to port: vLLM's image lanes are Buildkite pipeline
definitions, not a runtime suite, and its Python/PyTorch image layout has no
analogue here. The structural mirror is W4's workflow guard, exactly as
`ENG-RELEASE-BINARIES` mirrored the same pipeline with `check-release-workflow.py`
rather than porting a Python test.

The binding local executable specs are:

- `tests/scripts/test_server_package.py` — the staged-tree contract the image
  inherits; the image layout audit reuses its allowlist rather than restating it;
- `tests/scripts/test_release_manifest.py` — evidence-field semantics, reused
  unchanged for per-lane, per-architecture container evidence;
- the existing extracted-archive validation already invoked by the build
  scripts, which the image build inherits by construction; and
- new red-first mutation suites owed by this row: image layout (staged tree
  mutated, uid changed, `ffmpeg` removed, `VERSION`/tag mismatch), container
  smoke (endpoint removed, SIGTERM ignored), workflow guard (permission widened,
  push from a pull request, immutable tag reused, smoke step deleted), and the
  container-matrix checker (unpinned base, undeclared lane, missing retention).

Each mutation must fail for its own named reason. A suite that fails for the
wrong reason is not evidence that the gate works.

## Dependencies

- `ENG-RELEASE-BINARIES`' install/stage tree, build scripts and validator; this
  row consumes them and must not fork them;
- BuildKit with multi-platform support, and GitHub-hosted `ubuntu-latest` plus
  `ubuntu-24.04-arm` runners for native per-architecture builds;
- a CUDA toolkit image at a pinned digest for the cuda builder, and network
  access for `VLLM_CPP_CUTLASS_FETCH=ON` at a pinned revision;
- a software Vulkan ICD in the vulkan builder, because the build script runs
  `test_vulkan_backend` and `test_backend_cross_device`;
- GHCR package write authority on tags only, plus `id-token: write` scoped to
  the attestation job;
- redistribution-compatible licenses for the copied CUDA runtime libraries and
  for the distribution `ffmpeg` build; and
- matching hardware for any runtime claim: GB10 for arm64-cuda, and Thor for
  any Tegra determination, both under the project's GPU serialization protocol.

## Work breakdown

| Work | Deps | Deliverable | Exit gate |
|---|---|---|---|
| W1 (DONE) | — | `docker/Dockerfile` cpu target + builder, runtime stage, entrypoint/user/volumes/labels; `release/container-matrix.json` and its checker | local `docker build --target cpu` for the host arch; staged tree matches `package-server.py`'s allowlist; container smoke green; matrix checker red-first tested |
| W2 (DONE) | W1 | cuda and vulkan targets: builder bases, `JOBS` threading, CUDA runtime-lib copy and notices, BuildKit-safe driver stub, software ICD in the vulkan builder | both targets build locally; the release scripts' own validation passes inside the build; ten-SM `cuobjdump` inventory intact in the image |
| W3 (DONE) | W1 | image layout audit and container smoke as scripts with red-first mutation tests | mutating the staged tree, the uid, a missing `ffmpeg`, or a version mismatch each fail for the named reason |
| W4 (DONE) | W1, W2, W3 | `.github/workflows/containers.yml` with the seven stages, plus `scripts/check-container-workflow.py` and its mutation suite | dry run proves no push occurs; permission-widening, PR-push, and immutable-tag-reuse mutations red |
| W5 (DONE) | W4 | multi-arch manifest assembly, digest push, provenance attestation, two-part SBOM, retention | dry run produces both architecture digests and a manifest locally; attestation verifies against the manifest digest |
| W6 (OPEN) | W5 | matching-hardware runtime evidence: GB10 arm64-cuda real-model request; both-arch cpu baseline execution; Thor Tegra determination | per-tuple evidence recorded independently; no tuple inherits another's result |
| W7 (DONE except publication) | W6 | docs and first publication: `docs/USAGE.md` run recipes, `docs/FEATURES.md` lane surface, `README.md` quick start, `docs/STATUS.md` + `.agents/NOW.md` on the lifecycle move | published tags match the matrix; every documented command runs against the published bytes |


## Outcome

Measured on 2026-08-10, x86_64, Docker 29.1.2, from `docker/Dockerfile --target cpu`.

**The cpu lane works end to end.** A 783 MB image built from the release
scripts; `scripts/validate-container-image.py` passes config, layout and boot:
`/health` 200, `/version` 200, the image's own declared healthcheck passing
inside the container, and a clean SIGTERM. The boot leg ran against
`opt-125m-bf16-st`, so this is real runtime evidence for `linux/amd64` cpu and
for nothing else.

**The gate found a real product bug on its first run, which is the whole reason
it boots the server rather than running `--help` (issue #312).** `vllm-server`
installed no `SIGTERM` handler. As container PID 1 the kernel does not apply
default signal dispositions, so the signal was ignored outright: `docker stop`
waited its full 30 s grace and then `SIGKILL`ed, exit 137. Fixed by a self-pipe
handler that routes both `SIGTERM` and `SIGINT` into the same `server.stop()`
the existing `VT_BENCH_PROFILE_CONTROL` FIFO shutdown already used, installed at
all three `listen()` sites. **RED 137 after 30 s -> GREEN exit 0 in 0.25 s.**
Every orchestrator restart, rolling update and `compose down` was hard-killing
the server and dropping in-flight requests; it was invisible outside a container
because a non-PID-1 process dies on the default disposition anyway.

**Two build-context bugs, both silent.** `.dockerignore`'s `**/build*/` also
matched *files* -- Docker does not honour a trailing slash as directories-only
the way `.gitignore` does -- so `scripts/build-*-release.sh` was excluded from
the context and the build failed with `not found` after appearing to copy the
tree. And the builders needed `file` and `binutils`: the release scripts end in
`validate-release-archive.py`, which refuses to certify an archive it cannot
inspect, so a missing inspector failed the build *after* a full compile.

**What the design got right, and what it cost.** Calling the release scripts
rather than restating cmake flags meant the ten-SM list and the CPU tier gates
were never written twice, and the image inherited the extracted-archive audit
for free. The price was paid in builder dependencies -- qemu-user and a
SHA256-pinned Intel SDE for the CPU tier legs, a software Vulkan ICD because the
accelerator script *runs* `test_vulkan_backend` -- each of which is a real
requirement of the gate being inherited, not incidental weight. `-j 2` was
hardcoded in both scripts and is now a `JOBS` parameter.

**Hosted evidence.** `containers.yml` ran end to end on PR #307: `plan` green,
`verify (cpu, linux/amd64)` and `verify (vulkan, linux/amd64)` green -- the
vulkan lane's first build anywhere -- with `publish`, `manifest`, `attest` and
`promote` correctly skipped on a non-tag. CI has no model, so those runs cover
config and layout and report the absence of runtime evidence rather than
implying it.

**Not established, and not claimed.** No image is published. The `cuda` lane has
never been built, on this box or in CI: it is gated by the matrix checker and
the workflow guard, which prove structure, not that it compiles. The arm64 legs
are unbuilt, so the SBSA-vs-Tegra question (Thor `sm_110`, Orin `sm_87`) is
exactly as open as it was. The boot gate needs a model, so hosted CI runs config
and layout only and reports the absence of runtime evidence rather than
implying it -- W6 is where that closes.

### W6 GB10 result: the arm64 cuda lane runs on real silicon

Measured 2026-08-11 on `promaxgb10-4ad8` (GB10, `sm_121a`, aarch64, CUDA 13.3,
Docker 29.2.1 with the nvidia runtime), building `docker/Dockerfile --target
cuda` natively:

| axis | result |
|---|---|
| build | 673/673 objects; ten-SM gencode audit PASS; Triton AOT "six exact trees and namespaces OK" |
| image | **1.71 GB** (`linux/arm64`, cuda lane) |
| driver | `/usr/lib/aarch64-linux-gnu/libcuda.so.1 -> libcuda.so.580.159.03`, injected by the host runtime; the image ships none |
| runtime | `/health` 200, `/version` 200, the image's own declared healthcheck passing inside the container, clean SIGTERM -- **with `--gpus all`**, on `opt-125m-bf16-st` |

This is the first runtime evidence for any lane on accelerator hardware, and it
is evidence for exactly one tuple: `linux/arm64` + cuda + `sm_121a`. It says
nothing about amd64, and nothing about Tegra.

Getting here cost four defects, none of which existed in theory and all of which
the build found: the CUDA 12.9 base that could not compile `sm_110`, the
BuildKit cache mount that outlived its toolchain (both #366), the Marlin gencode
table drift that failed the audit on 14 correctly-compiled TUs (#394), and a
validator that could only ever produce build evidence because its boot smoke
never passed `--gpus`.

**SBSA is now confirmed, and it sharpens the Tegra question rather than
answering it.** The runtime libraries were copied from
`/usr/local/cuda/targets/sbsa-linux/lib` -- so the published arm64 image is an
SBSA image. GB10 runs it. Thor (`sm_110`) and Orin (`sm_87`) are Tegra/L4T with
a different CUDA runtime, remain unprobed, and are not covered by this result.

Lock discipline, since the box is shared: the build ran outside `gpu.lock`
because it needs no GPU, and only the container run took the lock, blocking --
it queued 16:58:38 -> 17:50:43 behind other users rather than jumping them.

### Pull-request scope, and why it is not a hole

A release run builds every lane on both architectures. A pull request builds a
reduced set -- `cpu` and `vulkan` on amd64 -- declared per lane in the matrix as
`verify_on_pull_request`. The reason is cost, not confidence: a ten-SM fat CUDA
image does not fit a hosted runner budget on every push, and a gate that
expensive becomes one people route around.

It is not a publication hole. `publish` consumes the FULL release matrix, and
rebuilds and revalidates each lane immediately before pushing it, so the worst
outcome a reduced pull-request matrix can produce is a failed release run --
never a published image that was not validated. `check-container-workflow.py`
enforces exactly that: the plan job must compute the publish matrix with
`--release`, and `publish` must consume `publish_matrix` rather than the reduced
one.

## Stop conditions

- The image and the published archive are ever described as the same bytes.
- Any lane grows a build definition that is not `scripts/build-*-release.sh`.
- A `latest-*` tag moves before both architectures are green.
- An immutable `:<version>-<lane>` tag is overwritten.
- Metal, MLX, or ROCm acquires an image without this spec changing first.
- A Tegra support claim is made from an SBSA build.
