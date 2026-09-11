ID: ISSUE-GH-968
Title: `windows-msvc-cpu`/`windows-msvc-vulkan` red on every branch based on `c7cb59fbb`: `C4244: conversion from 'const double' to 'float'` raised inside MSVC's `<vector>`/`<xutility>` from the two narrowing `positions.assign` calls [#964](https://github.com/mudler/vllm.cpp/pull/964) added at `ltx2_video.cpp:203,214`, where `StreamState::positions` and `Ltx2LatentState::positions` differ in element type. NOT fixed in flow: `#964`'s own comment at `:129-132` reasons that "double -> float -> double reproduces the bits", so the narrowing is DELIBERATE and a silencing cast is a claim about that reasoning rather than a formatting fix — it belongs to the LTX-2.5 lane. MATCHED-ARM evidence, by grepping each job log: #966 and #951 (both on `c7cb59fbb`) hit it, #967/#956/#950/#939/#938 (all pre-`c7cb59fbb`) do not. It was INVISIBLE until [#965](https://github.com/mudler/vllm.cpp/issues/965) removed the `C4456` shadow that failed the same jobs first — two independent causes stacked behind one habitually-red name, and neither was [#645](https://github.com/mudler/vllm.cpp/issues/645)
Row: MODEL-DIFFUSION-LTX25
State: UNKNOWN
Kind: bug
GitHub: 968
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:267`

### Frozen archive evidence

> | [#968](https://github.com/mudler/vllm.cpp/issues/968) | `MODEL-DIFFUSION-LTX25` | `windows-msvc-cpu`/`windows-msvc-vulkan` red on every branch based on `c7cb59fbb`: `C4244: conversion from 'const double' to 'float'` raised inside MSVC's `<vector>`/`<xutility>` from the two narrowing `positions.assign` calls [#964](https://github.com/mudler/vllm.cpp/pull/964) added at `ltx2_video.cpp:203,214`, where `StreamState::positions` and `Ltx2LatentState::positions` differ in element type. NOT fixed in flow: `#964`'s own comment at `:129-132` reasons that "double -> float -> double reproduces the bits", so the narrowing is DELIBERATE and a silencing cast is a claim about that reasoning rather than a formatting fix — it belongs to the LTX-2.5 lane. MATCHED-ARM evidence, by grepping each job log: #966 and #951 (both on `c7cb59fbb`) hit it, #967/#956/#950/#939/#938 (all pre-`c7cb59fbb`) do not. It was INVISIBLE until [#965](https://github.com/mudler/vllm.cpp/issues/965) removed the `C4456` shadow that failed the same jobs first — two independent causes stacked behind one habitually-red name, and neither was [#645](https://github.com/mudler/vllm.cpp/issues/645) | bug |

## Resolution

-
