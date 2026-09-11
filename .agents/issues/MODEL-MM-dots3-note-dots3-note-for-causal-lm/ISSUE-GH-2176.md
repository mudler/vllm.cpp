ID: ISSUE-GH-2176
Title: **dots3-note's nextn refusal is STRICTER than vLLM, which DROPS `model.layers.46.*` and `model.mtp.*` from the main model rather than refusing.** `Dots3NoteDeviceRefusal` turns away any config with `num_nextn_predict_layers > 0`, and §4 trap 3 correctly defaults that to 1 for a released `config.json` that does not carry the key — so every released checkpoint trips a branch upstream does not have. vLLM skips those weights in three places, re-derived at the row's pin `bc2d63e650`: `utils.py:542` `get_spec_layer_idx_from_weight_name` (matching `model.layers.{base+i}.` at `:559`), `deepseek_v2.py:1618-1620` `if spec_layer is not None: continue  # skip spec decode layers for main model`, and `models/dots3_note/nvidia/model.py:624` `if name.startswith("mtp."): continue` inside `Dots3NoteModel._adapt_weights`. `Dots3NoteLanguageModelForCausalLM` (`model.py:681`) subclasses `DeepseekV32ForCausalLM`, so the second is the path this architecture loads through. The repair is the classifier-deferral shape the vision and audio towers already use: a `nextn` bucket on `Dots3NoteAccounting`, filled by a `Dots3NoteIsNextnTensor(params, name)` predicate rather than a static-prefix table row because the prefix is config-derived, with the 19 tensors staying ENUMERATED so an absent one still refuses. Over the released index the split becomes 35362 language / 19 nextn / 2195 vision / 430 audio = 38006, against W2's 35381 / 2195 / 430. Fixed in flow with W5 (the MoE brick), because W5 is what makes the other half of the released config representable and the two together are what let `Dots3NoteDeviceRefusal(released_params)` return empty for the first time. Not the MTP head, which stays W10
Row: MODEL-MM-dots3-note-dots3-note-for-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 2176
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:847`

### Frozen archive evidence

> | [#2176](https://github.com/mudler/vllm.cpp/issues/2176) | `MODEL-MM-dots3-note` | **dots3-note's nextn refusal is STRICTER than vLLM, which DROPS `model.layers.46.*` and `model.mtp.*` from the main model rather than refusing.** `Dots3NoteDeviceRefusal` turns away any config with `num_nextn_predict_layers > 0`, and §4 trap 3 correctly defaults that to 1 for a released `config.json` that does not carry the key — so every released checkpoint trips a branch upstream does not have. vLLM skips those weights in three places, re-derived at the row's pin `bc2d63e650`: `utils.py:542` `get_spec_layer_idx_from_weight_name` (matching `model.layers.{base+i}.` at `:559`), `deepseek_v2.py:1618-1620` `if spec_layer is not None: continue  # skip spec decode layers for main model`, and `models/dots3_note/nvidia/model.py:624` `if name.startswith("mtp."): continue` inside `Dots3NoteModel._adapt_weights`. `Dots3NoteLanguageModelForCausalLM` (`model.py:681`) subclasses `DeepseekV32ForCausalLM`, so the second is the path this architecture loads through. The repair is the classifier-deferral shape the vision and audio towers already use: a `nextn` bucket on `Dots3NoteAccounting`, filled by a `Dots3NoteIsNextnTensor(params, name)` predicate rather than a static-prefix table row because the prefix is config-derived, with the 19 tensors staying ENUMERATED so an absent one still refuses. Over the released index the split becomes 35362 language / 19 nextn / 2195 vision / 430 audio = 38006, against W2's 35381 / 2195 / 430. Fixed in flow with W5 (the MoE brick), because W5 is what makes the other half of the released config representable and the two together are what let `Dots3NoteDeviceRefusal(released_params)` return empty for the first time. Not the MTP head, which stays W10 | bug |

## Resolution

-
