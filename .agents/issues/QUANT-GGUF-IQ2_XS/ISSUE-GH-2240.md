ID: ISSUE-GH-2240
Title: **IQ2_XS (17) and IQ4_XS (23) — the last two GGUF dequantizers the staged GLM-5.3-Flash artifact needed, and the two the loader stopped dead on.** "UD-Q2_K_XL" names a target average, not a format: of that artifact's 1412 tensors only TWO are Q2_K, while 82 are IQ2_XS (the `ffn_gate_exps`/`ffn_up_exps` routed experts) and 3 are IQ4_XS, so `LoadedEngine::FromModelDir` refused at `blk.3.ffn_gate_exps.weight has unknown ggml type id 17` before any dequant code ran — the reader had no block stride for 17, and the switch had no decoder for either. Both ported 1:1 from llama.cpp `b10451` (`ggml/src/ggml-quants.c:2516` `dequantize_row_iq2_xs`, `:2743` `dequantize_row_iq4_xs`, `ggml/src/ggml-common.h:627` `iq2xs_grid`) and gated BYTE-FOR-BYTE against the oracle's own decoders over REAL bytes read out of the two tensors that failed. IQ2_XS is the middle member of a family of three same-shaped codebooks — 256 / 512 / 1024 entries — where reaching for the wrong table still runs and still produces plausible magnitudes, so the 512-entry grid carries an FNV-1a seal as well. IQ4_XS reuses `kValuesIq4nl` unchanged; its delta is the super-block scale layout, a 6-bit `ls` spliced from a `scales_l` nibble and a `scales_h` bit pair and then biased by -32. Also carries the record correction the issue asked for: `.agents/specs/glm5-next-flash.md` O5/O8 are about the converter's WRITE side and were read as meaning the i-quant lane was absent entirely. Owning row `QUANT-GGUF-IQ2_XS` (and `QUANT-GGUF-IQ4_XS`); found by W5 of [#1998](https://github.com/mudler/vllm.cpp/issues/1998) via [#2223](https://github.com/mudler/vllm.cpp/issues/2223)
Row: QUANT-GGUF-IQ2_XS
State: UNKNOWN
Kind: feature
GitHub: 2240
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:869`

### Frozen archive evidence

> | [#2240](https://github.com/mudler/vllm.cpp/issues/2240) | `QUANT-GGUF-IQ2_XS` | **IQ2_XS (17) and IQ4_XS (23) — the last two GGUF dequantizers the staged GLM-5.3-Flash artifact needed, and the two the loader stopped dead on.** "UD-Q2_K_XL" names a target average, not a format: of that artifact's 1412 tensors only TWO are Q2_K, while 82 are IQ2_XS (the `ffn_gate_exps`/`ffn_up_exps` routed experts) and 3 are IQ4_XS, so `LoadedEngine::FromModelDir` refused at `blk.3.ffn_gate_exps.weight has unknown ggml type id 17` before any dequant code ran — the reader had no block stride for 17, and the switch had no decoder for either. Both ported 1:1 from llama.cpp `b10451` (`ggml/src/ggml-quants.c:2516` `dequantize_row_iq2_xs`, `:2743` `dequantize_row_iq4_xs`, `ggml/src/ggml-common.h:627` `iq2xs_grid`) and gated BYTE-FOR-BYTE against the oracle's own decoders over REAL bytes read out of the two tensors that failed. IQ2_XS is the middle member of a family of three same-shaped codebooks — 256 / 512 / 1024 entries — where reaching for the wrong table still runs and still produces plausible magnitudes, so the 512-entry grid carries an FNV-1a seal as well. IQ4_XS reuses `kValuesIq4nl` unchanged; its delta is the super-block scale layout, a 6-bit `ls` spliced from a `scales_l` nibble and a `scales_h` bit pair and then biased by -32. Also carries the record correction the issue asked for: `.agents/specs/glm5-next-flash.md` O5/O8 are about the converter's WRITE side and were read as meaning the i-quant lane was absent entirely. Owning row `QUANT-GGUF-IQ2_XS` (and `QUANT-GGUF-IQ4_XS`); found by W5 of [#1998](https://github.com/mudler/vllm.cpp/issues/1998) via [#2223](https://github.com/mudler/vllm.cpp/issues/2223) | feature |

## Resolution

-
