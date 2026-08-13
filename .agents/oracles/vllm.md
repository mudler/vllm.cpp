# vLLM — the primary oracle

The mirror source and the only oracle that outranks the others. Where vLLM
implements a behavior, it defines it; a secondary oracle disagreeing with vLLM
loses.

**The pin lives in [`../upstream-sync.md`](../upstream-sync.md)**, in its
` ```parity-pin ` block, together with the runtime and distribution version
strings a live oracle reports about itself and the `+g<sha>` constraint
`assert_oracle_commit` enforces. It is restated below only as identity — the
sync cycle advances the block over there, and `tools/bench/` reads that block,
not this file.

```oracle-pin
id = vllm
role = primary
upstream = https://github.com/vllm-project/vllm
scope = every behavior vLLM implements — defaults, modes, errors, edge cases, and both correctness and speed gates
pin = 5559679229bc961848b121ccdeaa8fa5d79bec98
pin_label = 0.26.0.dev0
pinned_on = 2026-07-26
gateable = yes
evidence = .agents/upstream-sync.md
```
