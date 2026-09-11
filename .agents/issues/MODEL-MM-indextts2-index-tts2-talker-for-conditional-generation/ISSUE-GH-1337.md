ID: ISSUE-GH-1337
Title: IndexTTS-2 drops `language` behind a 200. `SpeechGenParams::language` is documented as "upstream's `lang`; empty => the family's default" (`include/vllm/multimodal/speech_engine.h:76`), and `grep -rn '\.language' src/vllm/model_executor/models/` returns exactly ONE hit: `minimax_music3_speech.cpp:456`, which reads it only to REFUSE it. `grep language src/vllm/model_executor/models/indextts2.cpp` returns nothing, so `{"input":"ciao","language":"it"}` against an IndexTTS-2.5 server parses, reaches the family, is dropped, and returns audio in whatever language the model inferred, with no way for the caller to learn the key was ignored: [#925](https://github.com/mudler/vllm.cpp/issues/925) exactly, in the family that row did not touch. The model is NOT indifferent to the field — `indextts2_talker_loader.cpp:164` reads `lang_embedding.weight` into `out.languages`, so the checkpoint carries an embedding the request path never selects with. This also corrects a justification: `.agents/specs/minimax-music3.md` §17.4 argued `language` stays refused one layer down because "moving it up would break IndexTTS-2.5", which is not true today (no family reads it, so moving it up would convert this silent drop into a refusal); §17.4 now carries the forward-looking form instead. Two acceptable fixes: wire `language` into the talker, or refuse it by name at the IndexTTS-2 family layer until that lands. NOT fixed in [#1328](https://github.com/mudler/vllm.cpp/pull/1328), which is scoped to `ParseSpeechRequest`; either fix is IndexTTS-2 model code with its own oracle reading against the vLLM-Omni TTS lane
Row: MODEL-MM-indextts2-index-tts2-talker-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 1337
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:457`

### Frozen archive evidence

> | [#1337](https://github.com/mudler/vllm.cpp/issues/1337) | `MODEL-MM-indextts2-index-tts2-talker-for-conditional-generation` | IndexTTS-2 drops `language` behind a 200. `SpeechGenParams::language` is documented as "upstream's `lang`; empty => the family's default" (`include/vllm/multimodal/speech_engine.h:76`), and `grep -rn '\.language' src/vllm/model_executor/models/` returns exactly ONE hit: `minimax_music3_speech.cpp:456`, which reads it only to REFUSE it. `grep language src/vllm/model_executor/models/indextts2.cpp` returns nothing, so `{"input":"ciao","language":"it"}` against an IndexTTS-2.5 server parses, reaches the family, is dropped, and returns audio in whatever language the model inferred, with no way for the caller to learn the key was ignored: [#925](https://github.com/mudler/vllm.cpp/issues/925) exactly, in the family that row did not touch. The model is NOT indifferent to the field — `indextts2_talker_loader.cpp:164` reads `lang_embedding.weight` into `out.languages`, so the checkpoint carries an embedding the request path never selects with. This also corrects a justification: `.agents/specs/minimax-music3.md` §17.4 argued `language` stays refused one layer down because "moving it up would break IndexTTS-2.5", which is not true today (no family reads it, so moving it up would convert this silent drop into a refusal); §17.4 now carries the forward-looking form instead. Two acceptable fixes: wire `language` into the talker, or refuse it by name at the IndexTTS-2 family layer until that lands. NOT fixed in [#1328](https://github.com/mudler/vllm.cpp/pull/1328), which is scoped to `ParseSpeechRequest`; either fix is IndexTTS-2 model code with its own oracle reading against the vLLM-Omni TTS lane | bug |

## Resolution

-
