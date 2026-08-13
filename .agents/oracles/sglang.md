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

**Not gateable yet:** no SGLang run has been recorded on this project's hardware.
The parity-oracle spec is explicitly a read-only scoping spike ("no engine code,
no measurement taken here"), and the low-concurrency spec fixes the artifacts
"before anyone installs SGLang". Source has been read; nothing has been executed.

```oracle-pin
id = sglang
role = secondary
upstream = https://github.com/sgl-project/sglang
scope = a model or serving path SGLang implements and vLLM does not, plus the SGLang correctness cross-check and performance floor
pin = f63458b5beaceabbd9d749b9fc956370e1b649e6
pin_label = v0.5.15
pinned_on = 2026-07-27
gateable = no
evidence = #647
```
