ID: ISSUE-GH-925
Title: `POST /v1/audio/speech` silently ignored `audio_duration_s` — the name of the FIELD the key fills, and the spelling `speech_api.h:55` / `vllm.h:1032` / the C API all invite — so a request carrying it got a 200, a well-formed WAV, and the family's DEFAULT duration instead of the one it asked for. Every other unsupported field in `ParseSpeechRequest` is refused and named for exactly this reason; this key was the exception. It is the whole of [#852](https://github.com/mudler/vllm.cpp/issues/852): the e2e gate posted `audio_duration_s: 0.1`, ran 60 s instead (1500 AR frames not 2, 8 denoise windows not 1, 5167 vocoder latents not 6 — a ~750x job), and four runs were killed inside it and read as a hung weight load. FIXED IN FLOW: the near-miss is REFUSED, red-first in `test_speech_api.cpp`
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 925
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:206`

### Frozen archive evidence

> | [#925](https://github.com/mudler/vllm.cpp/issues/925) | `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation` | `POST /v1/audio/speech` silently ignored `audio_duration_s` — the name of the FIELD the key fills, and the spelling `speech_api.h:55` / `vllm.h:1032` / the C API all invite — so a request carrying it got a 200, a well-formed WAV, and the family's DEFAULT duration instead of the one it asked for. Every other unsupported field in `ParseSpeechRequest` is refused and named for exactly this reason; this key was the exception. It is the whole of [#852](https://github.com/mudler/vllm.cpp/issues/852): the e2e gate posted `audio_duration_s: 0.1`, ran 60 s instead (1500 AR frames not 2, 8 denoise windows not 1, 5167 vocoder latents not 6 — a ~750x job), and four runs were killed inside it and read as a hung weight load. FIXED IN FLOW: the near-miss is REFUSED, red-first in `test_speech_api.cpp` | bug |

## Resolution

-
