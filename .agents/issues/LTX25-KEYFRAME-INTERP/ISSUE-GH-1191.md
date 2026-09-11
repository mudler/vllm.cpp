ID: ISSUE-GH-1191
Title: `ltx2-gen` parsed `--first-frame` (`examples/ltx2_gen/main.cpp:333`) and assigned `vp.first_frame` (`:413`) while never reading `last_frame` at all, though `vllm_video_params` has carried the field since `include/vllm.h:948` and the LTX-2.5 engine has SERVED it since row `LTX25-TOKEN-APPEND` ([#930](https://github.com/mudler/vllm.cpp/issues/930)): the `wants_last_frame` arm of the phase loop places the image as a `VideoConditionByKeyframeIndex` at pixel frame `frames - 1`, gated by `test_ltx2_video`. So a closing keyframe was reachable from the C API and not from the shipped CLI. Latent until row `LTX25-KEYFRAME-INTERP` ([#1096](https://github.com/mudler/vllm.cpp/issues/1096)), which is the first caller it actually narrows: `KeyframeInterpolationPipeline` exists to generate the motion BETWEEN pinned keyframes, and with one slot the CLI can only ask for half of that. Fixed in flow by that row — one `--last-frame` flag parsed and assigned beside `--first-frame`, sharing the same `--image-crf` and strength the two slots already share, plus the usage and help text. An INTERIOR keyframe stays unrequestable and is [#1187](https://github.com/mudler/vllm.cpp/issues/1187)
Row: LTX25-KEYFRAME-INTERP
State: UNKNOWN
Kind: bug
GitHub: 1191
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:373`

### Frozen archive evidence

> | [#1191](https://github.com/mudler/vllm.cpp/issues/1191) | `LTX25-KEYFRAME-INTERP` | `ltx2-gen` parsed `--first-frame` (`examples/ltx2_gen/main.cpp:333`) and assigned `vp.first_frame` (`:413`) while never reading `last_frame` at all, though `vllm_video_params` has carried the field since `include/vllm.h:948` and the LTX-2.5 engine has SERVED it since row `LTX25-TOKEN-APPEND` ([#930](https://github.com/mudler/vllm.cpp/issues/930)): the `wants_last_frame` arm of the phase loop places the image as a `VideoConditionByKeyframeIndex` at pixel frame `frames - 1`, gated by `test_ltx2_video`. So a closing keyframe was reachable from the C API and not from the shipped CLI. Latent until row `LTX25-KEYFRAME-INTERP` ([#1096](https://github.com/mudler/vllm.cpp/issues/1096)), which is the first caller it actually narrows: `KeyframeInterpolationPipeline` exists to generate the motion BETWEEN pinned keyframes, and with one slot the CLI can only ask for half of that. Fixed in flow by that row — one `--last-frame` flag parsed and assigned beside `--first-frame`, sharing the same `--image-crf` and strength the two slots already share, plus the usage and help text. An INTERIOR keyframe stays unrequestable and is [#1187](https://github.com/mudler/vllm.cpp/issues/1187) | bug |

## Resolution

-
