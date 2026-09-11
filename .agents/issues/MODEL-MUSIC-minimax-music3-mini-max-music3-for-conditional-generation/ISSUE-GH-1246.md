ID: ISSUE-GH-1246
Title: MiniMax-Music3: the composed depth SCHEDULE in `Music3DepthStage` (`src/vllm/model_executor/models/minimax_music3_llm.cpp:398-535`) had NO gate — its only two call sites were its own definition and `minimax_music3_llm.cpp:617`, so deleting the 3-row prefix's position-0 K/V append, or silently dropping the fed-back `(index-1)*audio_vocab_size + drawn` projection row (which changes the generated song), each left ALL FIVE music3 suites GREEN. The row's answer was the FNV-1a fingerprint from `tools/bench/music3_depth_stage_ab.cpp`, but that file is a hand TRANSCRIPTION of the schedule rather than a call to `Music3DepthStage`, so it cannot detect divergence between itself and the function it transcribes — and no `CMakeLists.txt` compiled it either. FIXED IN FLOW: `test_minimax_music3_ar` gains a case driving the PRODUCTION function against a transcription of the whole-sequence schedule it replaced, at 8 heads of 8 / 2 layers / 8 codebooks / audio_vocab 32, checkpoint-free, 448 values bitwise plus the drawn codes, the draw count and its own teeth; both mutations now red it and it alone. The driver is compiled by CI as the never-linked OBJECT libraries `vllm_music3_depth_stage_ab_{before,after}`, which caught it failing `-Werror=comment` on the first build. Owned by `MUSIC3-DEPTH-SPEED`, spec §15.5.
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 1246
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:420`

### Frozen archive evidence

> | [#1246](https://github.com/mudler/vllm.cpp/issues/1246) | `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation` | MiniMax-Music3: the composed depth SCHEDULE in `Music3DepthStage` (`src/vllm/model_executor/models/minimax_music3_llm.cpp:398-535`) had NO gate — its only two call sites were its own definition and `minimax_music3_llm.cpp:617`, so deleting the 3-row prefix's position-0 K/V append, or silently dropping the fed-back `(index-1)*audio_vocab_size + drawn` projection row (which changes the generated song), each left ALL FIVE music3 suites GREEN. The row's answer was the FNV-1a fingerprint from `tools/bench/music3_depth_stage_ab.cpp`, but that file is a hand TRANSCRIPTION of the schedule rather than a call to `Music3DepthStage`, so it cannot detect divergence between itself and the function it transcribes — and no `CMakeLists.txt` compiled it either. FIXED IN FLOW: `test_minimax_music3_ar` gains a case driving the PRODUCTION function against a transcription of the whole-sequence schedule it replaced, at 8 heads of 8 / 2 layers / 8 codebooks / audio_vocab 32, checkpoint-free, 448 values bitwise plus the drawn codes, the draw count and its own teeth; both mutations now red it and it alone. The driver is compiled by CI as the never-linked OBJECT libraries `vllm_music3_depth_stage_ab_{before,after}`, which caught it failing `-Werror=comment` on the first build. Owned by `MUSIC3-DEPTH-SPEED`, spec §15.5. | bug |

## Resolution

-
