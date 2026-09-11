ID: ISSUE-GH-953
Title: `POST /v1/audio/speech` silently DROPPED five keys that SGLang-Omni, serving this same model on this same route, refuses BY NAME: `temperature`, `top_p`, `top_k`, `repetition_penalty` (`request_builders.py:14-19,109-114` — this model's AR stage has ONE sampler, a fixed top-50 draw, `encoders.py:48,94-103`, so the knobs can be neither honoured nor honestly ignored) and `max_new_tokens` (`request_builders.py:56-68` — upstream's LENGTH spelling, counted in 25 Hz FRAMES rather than seconds, so a 250-frame request silently became the family's 60 s default). The identical class as [#925](https://github.com/mudler/vllm.cpp/issues/925), which cost four multi-hour runs. FIXED IN FLOW while sweeping [#672](https://github.com/mudler/vllm.cpp/issues/672) for upstream parity: all five refused by name, RED first in `test_speech_api.cpp`, two mutations both firing
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 953
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:265`

### Frozen archive evidence

> | [#953](https://github.com/mudler/vllm.cpp/issues/953) | `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation` | `POST /v1/audio/speech` silently DROPPED five keys that SGLang-Omni, serving this same model on this same route, refuses BY NAME: `temperature`, `top_p`, `top_k`, `repetition_penalty` (`request_builders.py:14-19,109-114` — this model's AR stage has ONE sampler, a fixed top-50 draw, `encoders.py:48,94-103`, so the knobs can be neither honoured nor honestly ignored) and `max_new_tokens` (`request_builders.py:56-68` — upstream's LENGTH spelling, counted in 25 Hz FRAMES rather than seconds, so a 250-frame request silently became the family's 60 s default). The identical class as [#925](https://github.com/mudler/vllm.cpp/issues/925), which cost four multi-hour runs. FIXED IN FLOW while sweeping [#672](https://github.com/mudler/vllm.cpp/issues/672) for upstream parity: all five refused by name, RED first in `test_speech_api.cpp`, two mutations both firing | bug |

## Resolution

-
