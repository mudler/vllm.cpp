# Laguna (`LagunaForCausalLM`) — the registry forward routes to an arm no loader fills

**Row:** `MODEL-LAGUNA-REGISTRY-FORWARD-2618` — new row, `ACTIVE`.
**Issue:** [#2618](https://github.com/mudler/vllm.cpp/issues/2618).
**Date:** 2026-09-03. **Base:** `ca07f6e94`.
**Predecessor rows:** `MODEL-TEXT-laguna-laguna-for-causal-lm` (W5 landed the real
keep-quant forward), `ENG-ASYNC-DEVICE-IDS-2544` (whose `## Owed` O6 names this).

## Scope

`ModelRegistry::Forward` on `LagunaForCausalLM` must run the forward that serves
the checkpoints this tree's own loaders produce. Today it does not, on either
arm, and on one of them it does not fail loudly.

Out of scope: the paged/incremental arm (`LagunaForwardGgufCached`), any device
assembly, any throughput axis, and any change to `LagunaForwardGguf` itself.

## What is actually true (the issue is half right)

#2618 states "No loader in this tree fills those". Grounded against the tree at
`ca07f6e94`, the mechanism is **two different failures on two different arms**,
and the issue describes only the first:

| Arm | `moe.experts_gate/up/down` | `ReadF32` on it | Registry step today |
|---|---|---|---|
| safetensors NVFP4 (`LoadLagunaForCausalLMWeights`) | **EMPTY** — the loader fills `experts_gate_fp4` etc. instead (`laguna_weights.cpp:443-450`) and sets `has_nvfp4_weights = true` (`:473`) | reads an empty vector, then `LagunaModel::Forward` slices `exp_g.begin() + id*gu_stride` out of range | **SIGSEGV** — silent out-of-bounds copy |
| GGUF keep-quant (`LoadLagunaFromGgufShards`) | **FILLED**, as Q4_K/Q5_K block tensors (`laguna_weights.cpp:807-809`), `has_gguf_weights = true` (`:817`) | `ReadF32` (`laguna.cpp:172-190`) refuses a non-f32/bf16 dtype with `VT_CHECK(false, "laguna reference forward: weight dtype must be f32/bf16 ...")` | **throws**, and it throws EARLIER than the MoE block — `lw.attn.q_proj` is Q8_0, so the refusal fires at the first attention GEMM |

So the correction to the issue's text: the GGUF arm is filled and loud; only the
NVFP4 safetensors arm is unfilled and silent. The issue's stack trace is the
NVFP4 arm's, and its fixture (`test_laguna_nvfp4_loader.cpp::BuildFiniteTensors`)
is a safetensors fixture, which is consistent.

The rest of the issue is confirmed exactly:

- `ForwardLagunaForCausalLM` (`laguna_registry.cpp`) has one destination on both
  branches: `LagunaModel::ForwardDevice` (`laguna.cpp:1632`) itself calls
  `LagunaModel::Forward` (`laguna.cpp:1423`), the unit-gated f32 reference.
- `LagunaForwardGguf` (`laguna.cpp:1655`) has **no registry caller**.
  `grep -rn 'LagunaForwardGguf' src/ include/ tests/ examples/` returns its
  declaration, its definition, two comment mentions, `examples/laguna_gen`
  (an example, which §"Nothing lands dead" excludes as a production entry point)
  and `tests/vllm/models/test_laguna_nvfp4_loader.cpp` (a test, likewise
  excluded). The real forward is reachable from no production entry point.

## Design — three decisions, all settled by an in-tree precedent

#2618 says the fix needs three decisions. `deepseek_v4_registry.cpp:135-225` has
already made all three for the same shape of model (a reference `Forward`, a
`ForwardDevice` that delegates to it, and quantized towers the reference cannot
read). Mirroring it is therefore not a new product decision.

**D1 — which arm the registration selects.** The resident-quant arm first:
`weights.has_gguf_weights || weights.has_nvfp4_weights` routes to
`LagunaForwardGguf`. This is `LagunaForwardGguf`'s own precondition, quoted from
its `VT_CHECK` at `laguna.cpp:1667`, so the route predicate and the refusal
predicate are the same predicate rather than two copies that can drift.

**D2 — where the arm sits.** BEFORE the `input.gather_logits` test, exactly as
ds4 places its EXL3 paged arm and for the documented reason:
`ModelForwardInput::gather_logits` defaults to true and the runner leaves it true
on every default step, so a branch placed after that test is unreachable on a
default configuration. Placing the new arm after it would land it dead.

**D3 — what happens when neither flag is set.** Fall through to the existing f32
reference, exactly as ds4 falls through to `DeepseekV4Model::Forward`. No new
refusal is invented: a `LagunaWeights` with neither flag can only come from a
hand-built synthetic tower, which is what the reference exists to serve. Every
production loader sets one flag, so every production step takes the new arm.

## Risks

R1. `LagunaForwardGguf` is a stateless whole-sequence recompute and ignores
`attn_kv`. So does `LagunaModel::Forward` (`laguna.cpp:1429-1430` casts both
`attn_meta` and `attn_kv` to void). The change therefore does not regress the
cache contract; it moves from a forward that crashes to one that computes. The
incremental arm stays owed.

R2. The change makes the f32 reference unreachable from the registry. It remains
reachable from its unit tests, which is its role as the oracle for
`LagunaForwardGguf`. This is stated rather than hidden.

## Tests

`tests/vllm/models/test_laguna_registry_forward.cpp`, entering only through
`ModelRegistry::Load` + `ModelRegistry::Forward`:

- **T1** the registry step on the NVFP4 safetensors fixture returns
  `rows * vocab` finite logits. RED on `ca07f6e94` is a SIGSEGV. doctest's own
  signal handler catches it and reports
  `FATAL ERROR: test case CRASHED: SIGSEGV`, so the counts read
  `test cases: 5 | 4 passed | 1 failed | 2 skipped` and
  `assertions: 64 | 64 passed | 0 failed`, process rc 139. The `0 failed`
  assertion count is the tell that the case DIED rather than asserted, and the
  `2 skipped` are T2 and T3, which never ran because the process was gone.
- **T2** the registry step's logits are byte-identical to a direct
  `LagunaForwardGguf` call on the same weights and the same tokens. This is what
  proves WHICH forward the registry reached; T1 alone would pass on any forward
  that returns finite numbers.
- **T3** the routed experts are consumed through the registry: zeroing every
  routed expert's packed gate codes changes the registry's logits.

## Gates

```sh
cmake --build build -j 3 --target test_laguna_registry_forward test_laguna_nvfp4_loader
ctest --test-dir build -R 'laguna' --output-on-failure
scripts/agent-preflight.sh
```

## Evidence

Recorded in the pull request body: RED (both doctest counts and the process
signal), GREEN (both counts), each mutation with its rebuild rc and the restore
sha256, and the reachability mutation (deleting the new call site).

Six mutations were run, each rebuilt and each restored to sha256
`965db05a0a699065d597266d3ff90916a8a5617a50deb0b66367cdc6a24459c9`. Five were
detected. One (M5, zeroing `positions`) SURVIVED, and O3 below is why.

## Stop conditions

Stop and report `NEEDS_DECISION` if the resident-quant arm turns out to need a
paged cache the registry cannot supply, because that is a different row.

## Owed

- O1. The incremental/paged registry arm (`LagunaForwardGgufCached`) — the
  registry still recomputes the whole sequence every step. Tracked by
  [#2618](https://github.com/mudler/vllm.cpp/issues/2618)'s successor; this row
  does not close it.
- O2. A real-checkpoint registry step. This row gates on the synthetic NVFP4
  fixture only, because it is the only forward-runnable Laguna fixture in the
  tree.
- O3. That fixture is CONSTANT, and the row measured it rather than assuming it.
  `BuildFiniteTensors()` through `LagunaForwardGguf` returns `0.0444346` for all
  24 logits, unchanged by position (`{0,1,2}` vs `{0,0,0}` vs `{0,2000,4000}`,
  maxdiff 0), unchanged by token id (`{1,3,2}` vs `{5,5,5}`, maxdiff 0), and
  identical across rows. So no gate on this fixture can see a RoPE, causal-mask,
  position-plumbing or embedding-gather defect, and the pre-existing
  `finite + deterministic` run-gate holds trivially. Tracked by
  [#2834](https://github.com/mudler/vllm.cpp/issues/2834). This is why M5's
  wrong-`positions` mutation SURVIVED and is recorded as survived rather than
  quietly dropped: the fixture, not the route, is what fails to detect it.
