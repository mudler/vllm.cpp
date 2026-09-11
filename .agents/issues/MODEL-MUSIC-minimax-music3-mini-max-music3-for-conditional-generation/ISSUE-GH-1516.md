ID: ISSUE-GH-1516
Title: **The two-binary A/B guard `.agents/specs/minimax-music3.md` §16.6a drew from a VOID pair is unfalsifiable for an end-to-end pair.** `examples/CMakeLists.txt:425-426` links `minimax-music3-gen` against the SHARED `vllm::shared` (`CMakeLists.txt:2633`), so the timed program is a **72 744-byte ABI client** and the change under test lives in `libvllm_shared.so`, which no arm hashes. No `CMAKE_SKIP_BUILD_RPATH` is set, so two build directories put two RPATH strings into the client and the hashes differ whatever the source says — measured on `rc` job `c206ec87`, two clones, both files 72 744 bytes, different sha256. §16.6b reads exactly this as "the precondition this section exists to insist on"; its result stands on its BEHAVIOURAL control (`ar.depth_forward` 1414 -> 808 calls) and not on the hash. `scripts/music3-vocoder-conv-ab.sh` is unaffected: `CMakeLists.txt:2578-2579` links the STATIC `vllm`.
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 1516
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:540`

### Frozen archive evidence

> | [#1516](https://github.com/mudler/vllm.cpp/issues/1516) | `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation` | **The two-binary A/B guard `.agents/specs/minimax-music3.md` §16.6a drew from a VOID pair is unfalsifiable for an end-to-end pair.** `examples/CMakeLists.txt:425-426` links `minimax-music3-gen` against the SHARED `vllm::shared` (`CMakeLists.txt:2633`), so the timed program is a **72 744-byte ABI client** and the change under test lives in `libvllm_shared.so`, which no arm hashes. No `CMAKE_SKIP_BUILD_RPATH` is set, so two build directories put two RPATH strings into the client and the hashes differ whatever the source says — measured on `rc` job `c206ec87`, two clones, both files 72 744 bytes, different sha256. §16.6b reads exactly this as "the precondition this section exists to insist on"; its result stands on its BEHAVIOURAL control (`ar.depth_forward` 1414 -> 808 calls) and not on the hash. `scripts/music3-vocoder-conv-ab.sh` is unaffected: `CMakeLists.txt:2578-2579` links the STATIC `vllm`. | bug |

## Resolution

-
