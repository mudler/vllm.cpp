ID: ISSUE-GH-1151
Title: The `requires_distilled_lora` refusal still advertised [#1118](https://github.com/mudler/vllm.cpp/issues/1118) as OPEN, and cited `a2vid_two_stage.py`'s line numbers to every other pipeline. #1118 closed at `4ae0f54ab` (row `LTX25-PHASE-LORA`, PR [#1140](https://github.com/mudler/vllm.cpp/pull/1140)), which added `Ltx2PhaseRecipe::loras` and `Ltx2RebindDitLoras`; the message at `src/vllm/multimodal/ltx2_video.cpp:1039-1041` still ended "upstream fuses that adapter into stage 2 ALONE and this engine fuses once at load, so stage 1 sees it too", every clause of which had become false, and the comment above the refusal said the same. `ltx25-phase-lora.md` repaired the REFERENCE-CONDITIONING refusal, which carried the identical claim ~1100 lines away, and named only that one in its port map, so this site and the `ltx2-gen --help` text (`examples/ltx2_gen/main.cpp:210-212`) were both missed. Second defect at the same site: the refusal is deliberately keyed on the FLAG rather than on the kind string so the next recipe inherits it (the comment names #1093 and #1096 as waiting), yet its body interpolated `im.pipeline_kind` into the first sentence and hard-coded a2vid's `:164`, `:114`, `:107` into the rest — so the first arm to inherit it would be told its own name and then a different pipeline's source lines. `--distilled-lora required=True` lives on `default_2_stage_arg_parser` (`utils/args.py:1123`, `:1140-1155`), which all of these pipelines select, and that shared anchor is what the message now cites. Found and fixed IN FLOW by row `LTX25-TI2VID-RECIPE`, the second user of the flag; `tests/vllm/multimodal/test_ltx2_video.cpp` asserted the string `1118` was PRESENT and now asserts it is absent
Row: LTX25-TI2VID-RECIPE
State: UNKNOWN
Kind: bug
GitHub: 1151
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:356`

### Frozen archive evidence

> | [#1151](https://github.com/mudler/vllm.cpp/issues/1151) | `LTX25-TI2VID-RECIPE` | The `requires_distilled_lora` refusal still advertised [#1118](https://github.com/mudler/vllm.cpp/issues/1118) as OPEN, and cited `a2vid_two_stage.py`'s line numbers to every other pipeline. #1118 closed at `4ae0f54ab` (row `LTX25-PHASE-LORA`, PR [#1140](https://github.com/mudler/vllm.cpp/pull/1140)), which added `Ltx2PhaseRecipe::loras` and `Ltx2RebindDitLoras`; the message at `src/vllm/multimodal/ltx2_video.cpp:1039-1041` still ended "upstream fuses that adapter into stage 2 ALONE and this engine fuses once at load, so stage 1 sees it too", every clause of which had become false, and the comment above the refusal said the same. `ltx25-phase-lora.md` repaired the REFERENCE-CONDITIONING refusal, which carried the identical claim ~1100 lines away, and named only that one in its port map, so this site and the `ltx2-gen --help` text (`examples/ltx2_gen/main.cpp:210-212`) were both missed. Second defect at the same site: the refusal is deliberately keyed on the FLAG rather than on the kind string so the next recipe inherits it (the comment names #1093 and #1096 as waiting), yet its body interpolated `im.pipeline_kind` into the first sentence and hard-coded a2vid's `:164`, `:114`, `:107` into the rest — so the first arm to inherit it would be told its own name and then a different pipeline's source lines. `--distilled-lora required=True` lives on `default_2_stage_arg_parser` (`utils/args.py:1123`, `:1140-1155`), which all of these pipelines select, and that shared anchor is what the message now cites. Found and fixed IN FLOW by row `LTX25-TI2VID-RECIPE`, the second user of the flag; `tests/vllm/multimodal/test_ltx2_video.cpp` asserted the string `1118` was PRESENT and now asserts it is absent | bug |

## Resolution

-
