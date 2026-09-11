ID: ISSUE-GH-2064
Title: **`Qwen4ExpHfConfigFromGguf` and `ParseQwen4ExpParams` have never been COMPOSED**, and three keys go missing between them, each silent in a different way. `indexer_kv_heads` is absent and the QSA group is all-or-nothing, so NO real `qwen4exp` file parses at all ("QSA config is missing required fields: indexer_kv_heads"). `ple_layer_ids` is absent and its absence is LEGAL, so the model resolves an EMPTY PLE set — no n-gram table, no PLE layer, `number_of_conv_states()` reporting 1 where the architecture needs 3 — with nothing refusing; W6a left `qwen4exp.ple.layers` unmapped on the ground that neither file says which end the offset is on, and the converter says so in one line (`ple_layers = [i - 1 for i in hp["ple_layer_ids"]]`, llama.cpp #27742 head `035e2273`), so the GGUF key is ZERO-based. `ple_embed_dim` is absent and defaults to `hidden_size`, which is right on the released checkpoint by COINCIDENCE. Invisible to both existing gates because W6a's builds a config and never parses it while W1's parses a config.json and never builds one. Found and FIXED IN FLOW while writing W5a ([#2031](https://github.com/mudler/vllm.cpp/issues/2031)); mutations M11/M12/M13 in `test_qwen4_exp_gguf_weights.cpp` red on each half.
Row: MODEL-MM-QWEN4-EXP
State: UNKNOWN
Kind: bug
GitHub: 2064
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:815`

### Frozen archive evidence

> | [#2064](https://github.com/mudler/vllm.cpp/issues/2064) | `MODEL-MM-QWEN4-EXP` | **`Qwen4ExpHfConfigFromGguf` and `ParseQwen4ExpParams` have never been COMPOSED**, and three keys go missing between them, each silent in a different way. `indexer_kv_heads` is absent and the QSA group is all-or-nothing, so NO real `qwen4exp` file parses at all ("QSA config is missing required fields: indexer_kv_heads"). `ple_layer_ids` is absent and its absence is LEGAL, so the model resolves an EMPTY PLE set — no n-gram table, no PLE layer, `number_of_conv_states()` reporting 1 where the architecture needs 3 — with nothing refusing; W6a left `qwen4exp.ple.layers` unmapped on the ground that neither file says which end the offset is on, and the converter says so in one line (`ple_layers = [i - 1 for i in hp["ple_layer_ids"]]`, llama.cpp #27742 head `035e2273`), so the GGUF key is ZERO-based. `ple_embed_dim` is absent and defaults to `hidden_size`, which is right on the released checkpoint by COINCIDENCE. Invisible to both existing gates because W6a's builds a config and never parses it while W1's parses a config.json and never builds one. Found and FIXED IN FLOW while writing W5a ([#2031](https://github.com/mudler/vllm.cpp/issues/2031)); mutations M11/M12/M13 in `test_qwen4_exp_gguf_weights.cpp` red on each half. | bug |

## Resolution

-
