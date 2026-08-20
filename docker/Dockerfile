# syntax=docker/dockerfile:1.7
#
# vllm.cpp published container images -- ENG-RELEASE-CONTAINERS, issue #170.
# Design and boundaries: .agents/specs/container-images.md
#
# ONE file, three lane targets (`cpu`, `vulkan`, `cuda`). Every builder stage
# calls the SAME scripts/build-*-release.sh the release workflow calls, so the
# ten-SM gencode list, the CPU tier gates and the staged layout are written down
# exactly once. Those scripts already end in validate-release-archive.py, so an
# image build inherits the full archive audit rather than reimplementing it.
#
# The image IS the release bundle. It is NOT the same bytes as the published
# archive for the same version: both are built from the same definition, but
# separately, and each is attested on its own digest.
#
# Bases are pinned by digest. `docker buildx imagetools inspect <ref>` refreshes
# them; release/container-matrix.json records the same digests and is gated.
ARG UBUNTU_BASE=ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea
# CUDA 13.3, not 12.x: the ten-SM list includes sm_110 (Thor), and 12.9's
# nvcc rejects it outright -- `nvcc fatal: Unsupported gpu architecture
# 'compute_110'`. This must track the toolkit release.yml's CUDA job runs in
# (`container: nvidia/cuda:13.3.0-devel-ubuntu24.04`), because both build the
# same ten-SM fat binary from the same script.
ARG CUDA_BUILDER_BASE=nvidia/cuda:13.3.0-devel-ubuntu24.04@sha256:ef2203909e80b8b976cfc672f7e2ae2b00bc0e25c404ee86d89e10a3802f1c52

# ---------------------------------------------------------------------------
# Builder toolchains
# ---------------------------------------------------------------------------

FROM ${UBUNTU_BASE} AS builder-toolchain
ARG DEBIAN_FRONTEND=noninteractive
# `file`, `readelf` and `ldd` are not conveniences: the release scripts end in
# validate-release-archive.py, which refuses to certify an archive it cannot
# inspect. Without them the build fails late, after a full compile.
RUN apt-get update && apt-get install --yes --no-install-recommends \
      binutils \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      file \
      git \
      ninja-build \
      python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src

# The CPU lane's release script executes its compiled ISA tiers under emulation:
# a feature-poor leg that must run WITHOUT AVX2, and a feature-rich leg. That is
# scripts/run-cpu-release-gates.py, and it is the reason this builder carries
# emulators at all. Intel SDE is fetched by the repository's own installer,
# which pins the archive and checks its sha256; it never leaves this stage, so
# nothing about it is redistributed in a published image.
FROM builder-toolchain AS builder-cpu
ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH
RUN apt-get update && apt-get install --yes --no-install-recommends qemu-user \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/install-intel-sde.sh /tmp/install-intel-sde.sh
RUN if [ "$TARGETARCH" = "amd64" ]; then /tmp/install-intel-sde.sh /opt/intel-sde; fi

FROM builder-toolchain AS builder-vulkan
ARG DEBIAN_FRONTEND=noninteractive
# The accelerator script RUNS test_vulkan_backend and test_backend_cross_device.
# A builder with no ICD fails the build; it does not skip the test. lavapipe is
# the software ICD, and it is a BUILDER dependency only -- the runtime stage
# takes the loader and lets the host provide the real driver.
RUN apt-get update && apt-get install --yes --no-install-recommends \
      libvulkan-dev \
      libvulkan1 \
      mesa-vulkan-drivers \
      vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

FROM ${CUDA_BUILDER_BASE} AS builder-cuda
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --yes --no-install-recommends \
      binutils \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      file \
      git \
      ninja-build \
      python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src

# ---------------------------------------------------------------------------
# Lane builds. Each produces /out: the staged bundle package-server.py defines.
# ---------------------------------------------------------------------------

FROM builder-cpu AS build-cpu
ARG TARGETARCH
ARG VERSION
ARG SOURCE_SHA=unknown
ARG SOURCE_DATE_EPOCH=0
ARG EVIDENCE_URL=local-container-build
ARG JOBS=2
COPY . /src
RUN --mount=type=cache,target=/src/build-container-cpu,id=vllm-cpp-container-cpu-$TARGETARCH \
    set -eux; \
    if [ "$TARGETARCH" = "amd64" ]; then \
      artifact=linux-x86_64-glibc-cpu; arch=x86_64; \
      poor=$(command -v qemu-x86_64); rich=/opt/intel-sde/sde64; \
    else \
      artifact=linux-aarch64-glibc-cpu; arch=aarch64; \
      poor=$(command -v qemu-aarch64); rich=$poor; \
    fi; \
    VERSION="${VERSION:-$(sed -n 's/^project(vllm_cpp VERSION \([^ ]*\).*/\1/p' CMakeLists.txt)}" \
    SOURCE_SHA="$SOURCE_SHA" SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
    EVIDENCE_URL="$EVIDENCE_URL" JOBS="$JOBS" \
    docker/reset-stale-build-cache.sh build-container-cpu; \
    scripts/build-cpu-release.sh \
      "$artifact" "$arch" stable build-container-cpu \
      "$(getconf GNU_LIBC_VERSION | awk '{print $2}')" "$poor" "$rich"; \
    mkdir -p /out; cp -a build-container-cpu/release/stage/. /out/

FROM builder-vulkan AS build-vulkan
ARG TARGETARCH
ARG VERSION
ARG SOURCE_SHA=unknown
ARG SOURCE_DATE_EPOCH=0
ARG EVIDENCE_URL=local-container-build
ARG JOBS=2
COPY . /src
RUN --mount=type=cache,target=/src/build-container-vulkan,id=vllm-cpp-container-vulkan-$TARGETARCH \
    set -eux; \
    if [ "$TARGETARCH" = "amd64" ]; then artifact=linux-x86_64-glibc-vulkan; \
    else artifact=linux-aarch64-glibc-vulkan; fi; \
    VERSION="${VERSION:-$(sed -n 's/^project(vllm_cpp VERSION \([^ ]*\).*/\1/p' CMakeLists.txt)}" \
    SOURCE_SHA="$SOURCE_SHA" SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
    EVIDENCE_URL="$EVIDENCE_URL" JOBS="$JOBS" \
    docker/reset-stale-build-cache.sh build-container-vulkan; \
    scripts/build-linux-accelerator-release.sh "$artifact" vulkan build-container-vulkan; \
    mkdir -p /out; cp -a build-container-vulkan/release/stage/. /out/

FROM builder-cuda AS build-cuda
ARG TARGETARCH
ARG VERSION
ARG SOURCE_SHA=unknown
ARG SOURCE_DATE_EPOCH=0
ARG EVIDENCE_URL=local-container-build
ARG JOBS=2
COPY . /src
RUN --mount=type=cache,target=/src/build-container-cuda,id=vllm-cpp-container-cuda-$TARGETARCH \
    set -eux; \
    if [ "$TARGETARCH" = "amd64" ]; then artifact=linux-x86_64-glibc-cuda; \
    else artifact=linux-aarch64-glibc-cuda; fi; \
    VERSION="${VERSION:-$(sed -n 's/^project(vllm_cpp VERSION \([^ ]*\).*/\1/p' CMakeLists.txt)}" \
    SOURCE_SHA="$SOURCE_SHA" SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
    EVIDENCE_URL="$EVIDENCE_URL" JOBS="$JOBS" \
    docker/reset-stale-build-cache.sh build-container-cuda; \
    scripts/build-linux-accelerator-release.sh "$artifact" cuda build-container-cuda; \
    mkdir -p /out; cp -a build-container-cuda/release/stage/. /out/
# The CUDA runtime libraries the server links (CMakeLists.txt:1417) are
# redistributable under the CUDA EULA and are copied out of the toolkit. The
# DRIVER (libcuda.so.1) is deliberately absent: it belongs to the host and
# arrives through nvidia-container-toolkit.
RUN set -eux; mkdir -p /cudart; \
    for soname in libcudart.so libcublasLt.so libcublas.so; do \
      find /usr/local/cuda/targets/*/lib -name "$soname.*" -not -name '*stub*' \
        -exec cp -a {} /cudart/ \; ; \
    done; \
    cp -a /usr/local/cuda/targets/*/lib/libnvJitLink.so.* /cudart/ 2>/dev/null || true; \
    ls -1 /cudart

# ---------------------------------------------------------------------------
# Runtime. One base for every lane; each adds only what its backend resolves.
# ---------------------------------------------------------------------------

FROM ${UBUNTU_BASE} AS runtime-base
ARG DEBIAN_FRONTEND=noninteractive
# libssl3 is what `--model org/repo` needs: the server links libssl/libcrypto for
# the HuggingFace fetch (ENG-HF-MODEL-DOWNLOAD, #1280) and `huggingface.co`
# speaks nothing but https. `ca-certificates` below was already here and is the
# other half: without a trust store the handshake fails even with the library
# present. `scripts/validate-container-image.py` boots this image against an
# unknown repository and requires an HTTP STATUS FROM THE HUB, which is the only
# assertion that separates a working TLS build from one where the option
# silently resolved OFF. Measured 20 August 2026: the live hub answers 401, not
# 404, for an unknown repository asked anonymously.
# ffmpeg is a DELIBERATE divergence from the archive contract's no-vendor rule.
# An archive is extracted onto a host that has a PATH to inherit; a container
# has none, so the archive rule would ship a /v1/videos endpoint that can never
# succeed. Its licences are carried in the image SBOM and THIRD_PARTY_NOTICES.
RUN apt-get update && apt-get install --yes --no-install-recommends \
      ca-certificates \
      ffmpeg \
      libgomp1 \
      libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && userdel --remove ubuntu \
    && useradd --create-home --uid 1000 --user-group vllm \
    && mkdir -p /models /cache \
    && chown vllm:vllm /cache
ENV PATH=/opt/vllm/bin:$PATH \
    HF_HOME=/cache \
    XDG_CACHE_HOME=/cache
EXPOSE 8000
VOLUME ["/models", "/cache"]
USER 1000:1000
# The server already binds 0.0.0.0:8000 (server_main.cpp:135-136), the same
# default as vLLM, so there is no CMD: everything after the image name is a
# vllm-server flag.
ENTRYPOINT ["/opt/vllm/bin/vllm-server"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["/opt/vllm/bin/vllm-healthcheck"]

FROM runtime-base AS cpu
ARG VERSION
ARG SOURCE_SHA=unknown
COPY --from=build-cpu /out /opt/vllm
COPY docker/healthcheck.sh /opt/vllm/bin/vllm-healthcheck
LABEL org.opencontainers.image.title="vllm.cpp" \
      org.opencontainers.image.description="vllm.cpp OpenAI-compatible server, adaptive CPU lane" \
      org.opencontainers.image.source="https://github.com/mudler/vllm.cpp" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${SOURCE_SHA}" \
      io.vllm-cpp.lane="cpu" \
      io.vllm-cpp.channel="stable" \
      io.vllm-cpp.cpu-baseline="portable-sse2-without-avx2"

FROM runtime-base AS vulkan
ARG VERSION
ARG SOURCE_SHA=unknown
ARG DEBIAN_FRONTEND=noninteractive
USER root
# The loader only. The ICD and the device driver stay on the host; the image
# never claims to bundle a GPU driver.
RUN apt-get update && apt-get install --yes --no-install-recommends libvulkan1 \
    && rm -rf /var/lib/apt/lists/*
USER 1000:1000
COPY --from=build-vulkan /out /opt/vllm
COPY docker/healthcheck.sh /opt/vllm/bin/vllm-healthcheck
LABEL org.opencontainers.image.title="vllm.cpp" \
      org.opencontainers.image.description="vllm.cpp OpenAI-compatible server, Vulkan lane (preview)" \
      org.opencontainers.image.source="https://github.com/mudler/vllm.cpp" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${SOURCE_SHA}" \
      io.vllm-cpp.lane="vulkan" \
      io.vllm-cpp.channel="preview" \
      io.vllm-cpp.min-driver="host Vulkan 1.1 loader and ICD"

FROM runtime-base AS cuda
ARG VERSION
ARG SOURCE_SHA=unknown
COPY --from=build-cuda /cudart/ /opt/vllm/lib/
COPY --from=build-cuda /out /opt/vllm
COPY docker/healthcheck.sh /opt/vllm/bin/vllm-healthcheck
ENV LD_LIBRARY_PATH=/opt/vllm/lib
LABEL org.opencontainers.image.title="vllm.cpp" \
      org.opencontainers.image.description="vllm.cpp OpenAI-compatible server, fat CUDA lane (preview)" \
      org.opencontainers.image.source="https://github.com/mudler/vllm.cpp" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${SOURCE_SHA}" \
      io.vllm-cpp.lane="cuda" \
      io.vllm-cpp.channel="preview" \
      io.vllm-cpp.cuda-sms="80,86,87,89,90a,100a,103a,110,120a,121a" \
      io.vllm-cpp.min-driver="host NVIDIA driver providing libcuda.so.1 for CUDA 13.3"
