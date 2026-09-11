ID: ISSUE-GH-1338
Title: `speech_api.cpp`'s `audio_duration` refusal said "must be > 0" while the predicate was `>= 0.0` (`:215-216` @ `7d21644d7`), so an explicit `"audio_duration": 0` was ACCEPTED, resolved to the family's 60 s default, and the message that would have explained a refusal was printed by nothing. The CODE is the correct half: `minimax_music3_speech.cpp:465-474` documents `0` as "omitted, take the family default" and refuses only a NEGATIVE duration, and argues that split. So the parse-layer MESSAGE was the inaccurate one, and a caller who read it after a different failure learned a rule this route does not have. Same shape as [#925](https://github.com/mudler/vllm.cpp/issues/925). FIXED IN FLOW while repairing the review of [#1328](https://github.com/mudler/vllm.cpp/pull/1328), which was already editing the surrounding block: the message names NEGATIVE, and a red-first test pins both the accepted `0` and the message on a negative in both placements
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 1338
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:458`

### Frozen archive evidence

> | [#1338](https://github.com/mudler/vllm.cpp/issues/1338) | `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation` | `speech_api.cpp`'s `audio_duration` refusal said "must be > 0" while the predicate was `>= 0.0` (`:215-216` @ `7d21644d7`), so an explicit `"audio_duration": 0` was ACCEPTED, resolved to the family's 60 s default, and the message that would have explained a refusal was printed by nothing. The CODE is the correct half: `minimax_music3_speech.cpp:465-474` documents `0` as "omitted, take the family default" and refuses only a NEGATIVE duration, and argues that split. So the parse-layer MESSAGE was the inaccurate one, and a caller who read it after a different failure learned a rule this route does not have. Same shape as [#925](https://github.com/mudler/vllm.cpp/issues/925). FIXED IN FLOW while repairing the review of [#1328](https://github.com/mudler/vllm.cpp/pull/1328), which was already editing the surrounding block: the message names NEGATIVE, and a red-first test pins both the accepted `0` and the message on a negative in both placements | bug |

## Resolution

-
