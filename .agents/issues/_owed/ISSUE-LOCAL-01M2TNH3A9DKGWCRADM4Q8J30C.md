ID: ISSUE-LOCAL-01M2TNH3A9DKGWCRADM4Q8J30C
Title: The EXL3 head-to-head measures no prompt longer than 3.3k tokens, so every long-context claim stops at 40% of the served context
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-18
Updated: 2026-09-18
Closed: -

## Problem

The published head-to-head against exllamav3 stops at about 3.3k prompt tokens. The variadic corpus's longest band, `XL`, targets 9000 characters, and the realised histogram reads 2288 to 3290 prompt tokens (`docs/benchmarks/qwen38-27b-exl3-variadic-gb10.md`). Every long-context claim in this campaign therefore rests on prompts under half the served `--max-model-len 8192`.

That matters now, because #3150 reversed the prefill gap at the lengths we do measure. At c = 1 on `dgx:gpu0`, median TTFT is 1541 ms against their 2045 ms on the `L` band and 3648 ms against their 4389 ms on `XL`, where before #3150 the same bands read 3123 ms and 10300 ms. The mechanism (reconstruct + cuBLAS above 144 rows) amortizes further as M grows, so the lead should widen with length, and that is an untested prediction.

Two facts say the prediction must be measured rather than extrapolated:
- Acceptance falls with context on our side. Measured on `dgx:gpu0`, real HumanEval-shaped prompts: 0.49 at 324 input tokens, 0.45 at 2307, 0.37 at 8159, with drafted throughput 57.1, 52.6 and 42.0 tok/s (`docs/benchmarks/qwen38-27b-exl3-gb10.md`). Nothing measured the same axis on their engine.
- Their KV configuration is `-cs 262144` and ours auto-fits 8192, so a longer band also probes the part of the configuration this campaign has never matched.

The corpus generator has no band above `XL` (`benchmarks/variadic/build_corpus.py:31`, `XL_TARGET_CHARS = 9000`), so the workload cannot express the question.

## Resolution

-
