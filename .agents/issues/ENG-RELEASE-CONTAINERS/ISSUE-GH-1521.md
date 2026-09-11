ID: ISSUE-GH-1521
Title: **`docker/reset-stale-build-cache.sh` stamps only the compiler, CMake and CUDA, so a changed DEPENDENCY SET leaves a poisoned `CMakeCache.txt` in the BuildKit cache mount.** Measured 2026-08-20 at `d189f66dd` while verifying #1517. Two `cpu` lane image builds from one tree, differing only in whether the builder stages install `libssl-dev`, share `id=vllm-cpp-container-cpu-$TARGETARCH`. The first configured `-- Found OpenSSL: /usr/lib/x86_64-linux-gnu/libcrypto.so (found version "3.0.13")`. The second, with the package removed, should have taken the documented downgrade at `CMakeLists.txt:2479`; instead the retained cache still held `OPENSSL_CRYPTO_LIBRARY` and `OPENSSL_INCLUDE_DIR`, `find_package` reported found with an EMPTY version (`-- HuggingFace download: HTTPS through OpenSSL  (system, dynamic)`), and generate died with `Target "vllm" links to: OpenSSL::SSL but the target was not found`. The stamp is `c++ --version`, `cmake --version` and optionally `nvcc --version` plus the resolved `/usr/local/cuda`, none of which moved, so the reset never fired. LOWER severity than #1517 because the failure is LOUD and CI builds from a cold cache, but it silently invalidated the red arm of a red-to-green measurement and presented a third failure mode that reads like a code defect, which is the broken-instrument shape `.agents/verification.md` names. Worked around by giving the control build its own cache mount id. Found while fixing #1517 and deliberately not fixed in that flow: it moves cache-invalidation semantics and needs its own spec and fresh review
Row: ENG-RELEASE-CONTAINERS
State: UNKNOWN
Kind: bug
GitHub: 1521
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:537`

### Frozen archive evidence

> | [#1521](https://github.com/mudler/vllm.cpp/issues/1521) | `ENG-RELEASE-CONTAINERS` | **`docker/reset-stale-build-cache.sh` stamps only the compiler, CMake and CUDA, so a changed DEPENDENCY SET leaves a poisoned `CMakeCache.txt` in the BuildKit cache mount.** Measured 2026-08-20 at `d189f66dd` while verifying #1517. Two `cpu` lane image builds from one tree, differing only in whether the builder stages install `libssl-dev`, share `id=vllm-cpp-container-cpu-$TARGETARCH`. The first configured `-- Found OpenSSL: /usr/lib/x86_64-linux-gnu/libcrypto.so (found version "3.0.13")`. The second, with the package removed, should have taken the documented downgrade at `CMakeLists.txt:2479`; instead the retained cache still held `OPENSSL_CRYPTO_LIBRARY` and `OPENSSL_INCLUDE_DIR`, `find_package` reported found with an EMPTY version (`-- HuggingFace download: HTTPS through OpenSSL  (system, dynamic)`), and generate died with `Target "vllm" links to: OpenSSL::SSL but the target was not found`. The stamp is `c++ --version`, `cmake --version` and optionally `nvcc --version` plus the resolved `/usr/local/cuda`, none of which moved, so the reset never fired. LOWER severity than #1517 because the failure is LOUD and CI builds from a cold cache, but it silently invalidated the red arm of a red-to-green measurement and presented a third failure mode that reads like a code defect, which is the broken-instrument shape `.agents/verification.md` names. Worked around by giving the control build its own cache mount id. Found while fixing #1517 and deliberately not fixed in that flow: it moves cache-invalidation semantics and needs its own spec and fresh review | bug |

## Resolution

-
