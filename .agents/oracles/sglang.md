# SGLang — cross-check, performance floor, and vLLM-absent model paths

Two distinct roles, and one forbidden one — the methodology is
[`../specs/sglang-parity-oracle.md`](../specs/sglang-parity-oracle.md) and the
enumerated surface is [`../sglang-matrix.md`](../sglang-matrix.md):

- **Correctness cross-check.** Where SGLang and vLLM agree on greedy tokens for
  the same model, ours matches both. Where they diverge, **vLLM wins** — SGLang
  never re-specifies a vLLM-derived behavior.
- **Performance floor.** Wherever SGLang beats vLLM on an equivalent workload,
  SGLang is the binding floor.
- **Model paths vLLM does not implement at all.** This is the case the fallback
  rule was written for. Where that path is served by the SGLang-Omni pipeline
  runtime rather than SGLang proper — `MiniMax-Music3` is the first — the binding
  record is [`sglang-omni.md`](sglang-omni.md), not this one.
- **Forbidden:** porting SGLang's data structures as a second, incompatible
  abstraction. A SGLang-distinct behavior is an opt-in over our vLLM-derived
  design, never a fork of the engine.

**Two SGLang revisions already appear in specs, for two different purposes.**
The oracle pin below is the *source* pin — the tree every `file:line` in
`sglang-matrix.md` was read from. The benchmark harness in
[`../specs/cuda-sglang-low-concurrency.md`](../specs/cuda-sglang-low-concurrency.md)
separately pins tag `v0.5.13` / `28b095c01005d4a3a2a5b637b7d028b07fba31b2` and
its runtime image, and that pin is never silently substituted into the other's
evidence.

**SGLang-Omni is a third repository and has its own record**, since #672:
[`sglang-omni.md`](sglang-omni.md). This record covers SGLang proper. A model
present in one and absent from the other is the normal case, which is why the two
are not folded together.

**Gateable since 2026-07-28.** AGENTS.md sets the bar at "demonstrably builds and
runs the model", and this oracle cleared it under `CLAIM-SGLANG-PERF-BENCH`: the
`lmsysorg/sglang:v0.5.15-cu130@sha256:d0a667e` arm64 image was pulled and ran
`unsloth/Qwen3.6-27B-NVFP4` @ `890bdef7` on GB10 sm_121a with CUDA graphs
captured, needing no from-source build. Three repetitions at c8 and c16, both
arms driving the identical corpus and emitting exactly 80 by 128 output tokens
with zero errors. Numbers, corpus and teardown discipline are in
[`../sglang-matrix.md`](../sglang-matrix.md) under "Perf oracle results", and
`docs/STATUS.md` carries the same measurement.

This record said the opposite until 2026-08-16, for two and a half weeks after
the run. The two scoping specs it cited really were read-only, and neither was
revisited when the separate perf claim executed. Corrected under
[#979](https://github.com/mudler/vllm.cpp/issues/979), which discharges the
SGLang third of the three gateability debts
[#647](https://github.com/mudler/vllm.cpp/issues/647) holds open.

**Still open at this pin, and not covered by gateability:** the greedy token-ID
correctness cross-check (`SGLANG-ORACLE-CORRECT`, `INVENTORIED`), so SGLang binds
as a performance floor only for a model whose own correctness gate passed. The
c1, c2 and c4 points, the 35B arm and the shared-prefix cache-on arm are unrun.
No DSpark speculator ships in the pinned tree: `python/sglang/srt/speculative/` at
`f63458b5` carries DFlash, EAGLE, ngram and frozen-KV MTP and nothing named
`dspark`, so any drafted SGLang arm needs a deliberate pin advance first. That
claim is scoped to the directory on purpose. DSpark does NOT postdate this pin.
`docs_new/index.mdx:86,107,108,127` is tracked at `f63458b5` and links the
2026-07-06 lmsys blog announcing it, three days before the pinned commit's own
2026-07-09 date, and `speculative/spec_info.py:60-70` registers out-of-tree
algorithms at runtime, so absence from the directory listing is not absence at
runtime. This file said the pin predated DSpark until 2026-08-16. That wider
wording is withdrawn under
[#979](https://github.com/mudler/vllm.cpp/issues/979).

```oracle-pin
id = sglang
role = secondary
upstream = https://github.com/sgl-project/sglang
scope = a model or serving path SGLang implements and vLLM does not, plus the SGLang correctness cross-check and performance floor
pin = f63458b5beaceabbd9d749b9fc956370e1b649e6
pin_label = v0.5.15
pinned_on = 2026-07-27
gateable = yes
evidence = .agents/sglang-matrix.md
```
