ID: ISSUE-GH-1535
Title: **The C ABI version script is ELF-only, so the Apple leg has no equivalent guarantee.** `cmake/vllm_export.map` (`global: vllm_*; local: *;`) makes every httplib symbol inside `libvllm.so` local, which is what keeps `examples/video_studio/main.cpp` safe BY CONSTRUCTION even though it links `vllm::shared` (PRIVATE `vllm`, so no `CPPHTTPLIB_OPENSSL_SUPPORT`) and compiles the vendored header in the no-TLS layout while the library holds the TLS layout. The script is applied under `if(UNIX AND NOT APPLE)` (`CMakeLists.txt:2661`) because ld64 has no `--version-script`, so on macOS the dylib exports its C++ internals and whether Mach-O weak-def coalescing can bind the two layouts together is UNMEASURED. `capi_shared_exports_only_abi` is gated on the same condition and does not cover it. Low severity, no macOS host here; found while fixing [#1531](https://github.com/mudler/vllm.cpp/issues/1531)
Row: SERVE-C-ABI
State: UNKNOWN
Kind: bug
GitHub: 1535
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:542`

### Frozen archive evidence

> | [#1535](https://github.com/mudler/vllm.cpp/issues/1535) | `SERVE-C-ABI` | **The C ABI version script is ELF-only, so the Apple leg has no equivalent guarantee.** `cmake/vllm_export.map` (`global: vllm_*; local: *;`) makes every httplib symbol inside `libvllm.so` local, which is what keeps `examples/video_studio/main.cpp` safe BY CONSTRUCTION even though it links `vllm::shared` (PRIVATE `vllm`, so no `CPPHTTPLIB_OPENSSL_SUPPORT`) and compiles the vendored header in the no-TLS layout while the library holds the TLS layout. The script is applied under `if(UNIX AND NOT APPLE)` (`CMakeLists.txt:2661`) because ld64 has no `--version-script`, so on macOS the dylib exports its C++ internals and whether Mach-O weak-def coalescing can bind the two layouts together is UNMEASURED. `capi_shared_exports_only_abi` is gated on the same condition and does not cover it. Low severity, no macOS host here; found while fixing [#1531](https://github.com/mudler/vllm.cpp/issues/1531) | bug |

## Resolution

-
