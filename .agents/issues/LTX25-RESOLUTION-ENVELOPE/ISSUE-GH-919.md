ID: ISSUE-GH-919
Title: `vllm_video_generate` integer-divides `width`/`height` into the latent grid (`ltx2_video.cpp:1456-1463 @ 5a0ffe9e3`) with no divisibility check, so a 100x100 request on the distilled two-stage recipe silently renders 96x96. The only geometry guard in the LTX path is a LOWER bound (`ltx2_video.cpp:1464-1471 @ 5a0ffe9e3`). Every repo-local anchor in this row is SHA-pinned because the fix edits the files it cites, and inserts lines above both spans (#911). Upstream hard-validates and raises at the top of a pipeline `__call__` — `assert_resolution` (`ltx-pipelines utils/helpers.py:540-551` @ `fd4ded7f2`), 64 for two-stage and 32 for one-stage, NINE invocations including `ti2vid_two_stages.py:184` and `ti2vid_two_stages_hq.py:199` — so mirroring means refusing, not flooring. Nine, not the 21 lines a grep for the name returns (9 invocations + 1 definition + 10 imports + 1 `__all__` string), and not every pipeline: 13 pipeline `__call__`s take a resolution and the three `*_mgpu.py` variants plus `hdr_ic_lora.py:352` skip the guard. `docs/USAGE.md:626-629 @ 5a0ffe9e3` already documents the rule as though it were enforced. Frames are the OPPOSITE answer: upstream floors an explicit `num_frames` exactly as we do (`ltx_core/types.py:113`) and validates it nowhere, so that half is a doc correction
Row: LTX25-RESOLUTION-ENVELOPE
State: UNKNOWN
Kind: bug
GitHub: 919
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:247`

### Frozen archive evidence

> | [#919](https://github.com/mudler/vllm.cpp/issues/919) | `LTX25-RESOLUTION-ENVELOPE` | `vllm_video_generate` integer-divides `width`/`height` into the latent grid (`ltx2_video.cpp:1456-1463 @ 5a0ffe9e3`) with no divisibility check, so a 100x100 request on the distilled two-stage recipe silently renders 96x96. The only geometry guard in the LTX path is a LOWER bound (`ltx2_video.cpp:1464-1471 @ 5a0ffe9e3`). Every repo-local anchor in this row is SHA-pinned because the fix edits the files it cites, and inserts lines above both spans (#911). Upstream hard-validates and raises at the top of a pipeline `__call__` — `assert_resolution` (`ltx-pipelines utils/helpers.py:540-551` @ `fd4ded7f2`), 64 for two-stage and 32 for one-stage, NINE invocations including `ti2vid_two_stages.py:184` and `ti2vid_two_stages_hq.py:199` — so mirroring means refusing, not flooring. Nine, not the 21 lines a grep for the name returns (9 invocations + 1 definition + 10 imports + 1 `__all__` string), and not every pipeline: 13 pipeline `__call__`s take a resolution and the three `*_mgpu.py` variants plus `hdr_ic_lora.py:352` skip the guard. `docs/USAGE.md:626-629 @ 5a0ffe9e3` already documents the rule as though it were enforced. Frames are the OPPOSITE answer: upstream floors an explicit `num_frames` exactly as we do (`ltx_core/types.py:113`) and validates it nowhere, so that half is a doc correction | bug |

## Resolution

-
