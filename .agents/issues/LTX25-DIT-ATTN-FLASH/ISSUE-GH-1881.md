ID: ISSUE-GH-1881
Title: **The LTX-2.5 pixel A/B recorded a 75 KB LAUNCHER as its binary identity, and two builds a whole release window apart printed the SAME value while the 92 MB library that holds every measured kernel differed by 6,372,624 bytes.** Observed live on `dgx:gpu0` 2026-08-24 while the FA-2 ladder of [#1855](https://github.com/mudler/vllm.cpp/issues/1855) was building. `BINSHA=$(sha256sum "$BIN/ltx2-gen")` hashes 75,344 bytes of `main()`; `vt::Attention`, `vt::AttentionDenseFlash`, `vt::AttentionDenseFa2`, both VAEs and the loader are all in `libvllm.so.0.0.3`, which was hashed NOWHERE. Run `1612-r3` (source `3e2961ef0`, `libvllm` 85,703,328 B) and run `1853-fa2-r1` (source `62cbae10d`, `libvllm` `f046e75dcede2586...`, 92,075,952 B) both recorded `binary_sha256=834cec557c16cf77...`, each `binary_built=in-lease` with `BUILD_RC=0`. The launcher's own translation unit did not change, so its output is reproducible BY CONSTRUCTION - the one artefact whose hash was stable was the one containing none of the code under measurement. Consequence: §10.7's "the binary `834cec55...`", which #1743 and #1855 rest on, does NOT pin the code that produced them, and a later reader reads the same string and concludes the same code ran. **NOT an invalidation of 1612-r3**: its four arms ran from one build in one lease and each proved its own op from its own log; what its RECORD cannot do is tell its build from a later one. **FIXED IN FLOW for the pixel harness** - `LIBSHA` is computed, printed, written to `PROVENANCE` as `library_sha256` and added to every arm's `render.log` header, with three tripwires in `test_ltx25_pixel_ab_harness.py` that each red when their site is deleted. **STILL OWED**: `ltx25-dit-attn-flash-ab.sh` and `ltx25-dit-attn-fa2-hd128-ab.sh` carry the identical idiom, so every speed number they have recorded has the same hole. Owned by row `LTX25-DIT-ATTN-FLASH` and listed under `## Owed`
Row: LTX25-DIT-ATTN-FLASH
State: UNKNOWN
Kind: bug
GitHub: 1881
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:714`

### Frozen archive evidence

> | [#1881](https://github.com/mudler/vllm.cpp/issues/1881) | `LTX25-DIT-ATTN-FLASH` | **The LTX-2.5 pixel A/B recorded a 75 KB LAUNCHER as its binary identity, and two builds a whole release window apart printed the SAME value while the 92 MB library that holds every measured kernel differed by 6,372,624 bytes.** Observed live on `dgx:gpu0` 2026-08-24 while the FA-2 ladder of [#1855](https://github.com/mudler/vllm.cpp/issues/1855) was building. `BINSHA=$(sha256sum "$BIN/ltx2-gen")` hashes 75,344 bytes of `main()`; `vt::Attention`, `vt::AttentionDenseFlash`, `vt::AttentionDenseFa2`, both VAEs and the loader are all in `libvllm.so.0.0.3`, which was hashed NOWHERE. Run `1612-r3` (source `3e2961ef0`, `libvllm` 85,703,328 B) and run `1853-fa2-r1` (source `62cbae10d`, `libvllm` `f046e75dcede2586...`, 92,075,952 B) both recorded `binary_sha256=834cec557c16cf77...`, each `binary_built=in-lease` with `BUILD_RC=0`. The launcher's own translation unit did not change, so its output is reproducible BY CONSTRUCTION - the one artefact whose hash was stable was the one containing none of the code under measurement. Consequence: §10.7's "the binary `834cec55...`", which #1743 and #1855 rest on, does NOT pin the code that produced them, and a later reader reads the same string and concludes the same code ran. **NOT an invalidation of 1612-r3**: its four arms ran from one build in one lease and each proved its own op from its own log; what its RECORD cannot do is tell its build from a later one. **FIXED IN FLOW for the pixel harness** - `LIBSHA` is computed, printed, written to `PROVENANCE` as `library_sha256` and added to every arm's `render.log` header, with three tripwires in `test_ltx25_pixel_ab_harness.py` that each red when their site is deleted. **STILL OWED**: `ltx25-dit-attn-flash-ab.sh` and `ltx25-dit-attn-fa2-hd128-ab.sh` carry the identical idiom, so every speed number they have recorded has the same hole. Owned by row `LTX25-DIT-ATTN-FLASH` and listed under `## Owed` | bug |

## Resolution

-
