ID: ISSUE-GH-904
Title: `main` red on `sanitize-cpu (address,undefined)` for a THIRD reason after the two [#730](https://github.com/mudler/vllm.cpp/issues/730) enumerates, both of which are now closed: `~Ltx2DitDeviceWeights` frees the staged DiT buffers on the main thread (`ltx2_device.cpp:1088`) while a `vt::cpu` threadpool worker is still reading one inside `AddKernel` (`cpu_layernorm.cpp:33`), so the staged weights' lifetime is not joined to the in-flight parallel op that reads them. Deterministic, 5 runs / 5 aborts on unmodified `e8048ef63`; the rest of the suite is 477/478 and `test_ltx2_video` now PASSES
Row: ROAD-V1-LTX25
State: UNKNOWN
Kind: bug
GitHub: 904
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:239`

### Frozen archive evidence

> | [#904](https://github.com/mudler/vllm.cpp/issues/904) | `ROAD-V1-LTX25` | `main` red on `sanitize-cpu (address,undefined)` for a THIRD reason after the two [#730](https://github.com/mudler/vllm.cpp/issues/730) enumerates, both of which are now closed: `~Ltx2DitDeviceWeights` frees the staged DiT buffers on the main thread (`ltx2_device.cpp:1088`) while a `vt::cpu` threadpool worker is still reading one inside `AddKernel` (`cpu_layernorm.cpp:33`), so the staged weights' lifetime is not joined to the in-flight parallel op that reads them. Deterministic, 5 runs / 5 aborts on unmodified `e8048ef63`; the rest of the suite is 477/478 and `test_ltx2_video` now PASSES | bug |

## Resolution

-
