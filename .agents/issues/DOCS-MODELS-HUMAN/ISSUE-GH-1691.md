ID: ISSUE-GH-1691
Title: **`docs/QUICKSTART.md` said no container lane had ever been published, and three had.** The page carried `The container lanes have never been published, so no tag below resolves against the registry yet` and, on its `docker run` line, `The package carries only a stage tag today, which is a build artifact and not a lane`. Both were true when written and false on 2026-08-22: `ghcr.io/mudler/vllm.cpp` is public and carries `main-cpu`, `main-cuda` and `main-vulkan`, each a `linux/amd64` + `linux/arm64` manifest, published 07:30 UTC that day. `docs/guides/container-images.md` already documented the `:main-<lane>` tags, so the two pages disagreed. VERIFIED by execution rather than by reading the registry listing: `docker run --rm ghcr.io/mudler/vllm.cpp:main-cpu --version` answered `vllm.cpp 0.0.3 c-abi=23` at digest `sha256:7f88301ea282dad778748929e7aa6869d2418c8d295eef0e7900cca8310d06e5`, and the same image with a mounted `Qwen/Qwen3-0.6B` returned tokens through `/v1/completions` on host `mudler-ubuntu-box` (x86_64, Docker 29.1.2). The image also parses the `vllm_cpp` weight-residency document and echoes `mmap=on prefault=off expert_stream=on expert_stream_slots=4000`, which is what lets the Qwen3.8 2.4T page carry a container form of its recipe. FIXED IN FLOW: the false note is corrected, the `docker run` line names a tag that resolves, and the executed-row table gains its first real row. **This does NOT close [#1281](https://github.com/mudler/vllm.cpp/issues/1281)**: `:latest` still does not exist, `--model org/repo` is still blocked by [#1511](https://github.com/mudler/vllm.cpp/issues/1511) so the executed row mounts a local directory, and no GPU-lane row was run. The executed-row obligation stays owned by #1281
Row: DOCS-MODELS-HUMAN
State: UNKNOWN
Kind: bug
GitHub: 1691
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:630`

### Frozen archive evidence

> | [#1691](https://github.com/mudler/vllm.cpp/issues/1691) | `DOCS-MODELS-HUMAN` | **`docs/QUICKSTART.md` said no container lane had ever been published, and three had.** The page carried `The container lanes have never been published, so no tag below resolves against the registry yet` and, on its `docker run` line, `The package carries only a stage tag today, which is a build artifact and not a lane`. Both were true when written and false on 2026-08-22: `ghcr.io/mudler/vllm.cpp` is public and carries `main-cpu`, `main-cuda` and `main-vulkan`, each a `linux/amd64` + `linux/arm64` manifest, published 07:30 UTC that day. `docs/guides/container-images.md` already documented the `:main-<lane>` tags, so the two pages disagreed. VERIFIED by execution rather than by reading the registry listing: `docker run --rm ghcr.io/mudler/vllm.cpp:main-cpu --version` answered `vllm.cpp 0.0.3 c-abi=23` at digest `sha256:7f88301ea282dad778748929e7aa6869d2418c8d295eef0e7900cca8310d06e5`, and the same image with a mounted `Qwen/Qwen3-0.6B` returned tokens through `/v1/completions` on host `mudler-ubuntu-box` (x86_64, Docker 29.1.2). The image also parses the `vllm_cpp` weight-residency document and echoes `mmap=on prefault=off expert_stream=on expert_stream_slots=4000`, which is what lets the Qwen3.8 2.4T page carry a container form of its recipe. FIXED IN FLOW: the false note is corrected, the `docker run` line names a tag that resolves, and the executed-row table gains its first real row. **This does NOT close [#1281](https://github.com/mudler/vllm.cpp/issues/1281)**: `:latest` still does not exist, `--model org/repo` is still blocked by [#1511](https://github.com/mudler/vllm.cpp/issues/1511) so the executed row mounts a local directory, and no GPU-lane row was run. The executed-row obligation stays owned by #1281 | bug |

## Resolution

-
