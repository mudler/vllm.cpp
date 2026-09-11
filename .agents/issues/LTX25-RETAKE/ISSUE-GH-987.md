ID: ISSUE-GH-987
Title: Two `ltx-2.5` refusal messages state reasons that are no longer true. (a) `src/vllm/multimodal/ltx2_video.cpp:1608 @ 0e1bee42f` says "nothing reads `ref_video_dir` at all", and MiniMax-H3 has always consumed the directory in full (`ReadReferenceClipChw`, `src/vllm/multimodal/minimax_h3_video.cpp:135 @ 0e1bee42f`, called at `:650`); [#975](https://github.com/mudler/vllm.cpp/issues/975) inherited the wider claim from this message. The claim that holds is narrower: the LTX-2.5 engine never reads the directory's CONTENTS. (b) `ltx2_video.cpp:1636-1638 @ 0e1bee42f` says "there is no AUDIO_VAE_ENCODER key filter", and `c2019b0e3` landed `Ltx2AudioVaeEncoderKeyRules()` (`include/vllm/model_executor/models/ltx2_audio_input.h:73 @ 0e1bee42f`) with a live call through `Ltx2EncodeAudioToLatent`. Both rewritten in the `WHAT IS *NOT* THE REASON` shape in the same flow, with one assertion tied to the LOCAL fact that the LTX side now reads the directory
Row: LTX25-RETAKE
State: UNKNOWN
Kind: bug
GitHub: 987
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:274`

### Frozen archive evidence

> | [#987](https://github.com/mudler/vllm.cpp/issues/987) | `LTX25-RETAKE` | Two `ltx-2.5` refusal messages state reasons that are no longer true. (a) `src/vllm/multimodal/ltx2_video.cpp:1608 @ 0e1bee42f` says "nothing reads `ref_video_dir` at all", and MiniMax-H3 has always consumed the directory in full (`ReadReferenceClipChw`, `src/vllm/multimodal/minimax_h3_video.cpp:135 @ 0e1bee42f`, called at `:650`); [#975](https://github.com/mudler/vllm.cpp/issues/975) inherited the wider claim from this message. The claim that holds is narrower: the LTX-2.5 engine never reads the directory's CONTENTS. (b) `ltx2_video.cpp:1636-1638 @ 0e1bee42f` says "there is no AUDIO_VAE_ENCODER key filter", and `c2019b0e3` landed `Ltx2AudioVaeEncoderKeyRules()` (`include/vllm/model_executor/models/ltx2_audio_input.h:73 @ 0e1bee42f`) with a live call through `Ltx2EncodeAudioToLatent`. Both rewritten in the `WHAT IS *NOT* THE REASON` shape in the same flow, with one assertion tied to the LOCAL fact that the LTX side now reads the directory | bug |

## Resolution

-
