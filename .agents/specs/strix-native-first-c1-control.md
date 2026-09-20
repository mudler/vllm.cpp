# Isolate the native first-c1 inference fault

Row: `BACKEND-GATE-ROCM-SGLANG`.
Issue: [#3111](https://github.com/mudler/vllm.cpp/issues/3111).
Parent: [Strix performance](strix-qwen3-4b-c4-performance.md), #3076.
Dependency: [diagnostic publication](strix-lifecycle-diagnostic-publication.md), #3110.

## Now

This specification defines a diagnostic control, not a GPU fault repair.
No root cause or performance advantage is established.
The developer approved this bounded control on 9 September 2026 in
[#3111, comment 5602300636](https://github.com/mudler/vllm.cpp/issues/3111#issuecomment-5602300636).
The approval retains unchanged binaries, six prompts, 128 tokens, a 15-minute lease, and no requested reset.
It grants no SGLang policy change or correctness waiver.
Specification commit `0a5399ad09b49d5e7b4576daee698bd449b1b104` precedes implementation.
The implementation imports reviewed #3110 head `656ce703e6c48794c84d608733fbdba4c3e4461b` as a merge parent.
Fresh review, the operator's gate, and the bounded hardware control remain pending.

## Evidence and boundaries

Job `cfebd7eb-2612-455c-bd67-32bd28b5c800` ran the existing six-request qualification.
Pinned production vLLM completed its c1 and c4 corpora.
Its second c1 corpus differed from its first corpus.
The native adapter then returned successful configuration, including canonical prompt IDs.
Its first c1 corpus failed before the aggregate run reply.
The adapter exited with status `-6` and reported `GPU Hang`.

The earlier description of a configure failure was incorrect.
Configuration succeeded. The evidence does not identify the first failing request or token.
The operator cancelled the job after observing the fault.
Its terminal exit was 143 at 09:02:07 UTC on 9 September 2026.
Later comparator output is not accepted.

Kernel messages report failed MES queue removal and an automatic GPU reset.
The reset started at 09:01:15 UTC and succeeded at 09:01:18 UTC.
The kernel reported lost VRAM contents.
The operator did not request a reset or clear an unhealthy device.
Read-only health job `24c2b5d4-61f0-4677-b417-0be72403c6b8` later enumerated `gfx1151`.
ROCm reported an idle device. The health checks produced no new kernel fault.
These observations do not prove reliable inference.

Evidence resides in campaign `strix-four-engine-3053.X94a3J` on the resolved NAS:
`child3108-production-qualification.jGml1g`,
`native-hang-health.6z03Bv`, and `native-hang-sysfs.oXwOaZ`.
These paths identify evidence, not another developer's environment defaults.

The model is Qwen/Qwen3-4B BF16 at revision
`1cfa9a7208912126459214e8b04321603b3df60c`.
The unchanged `qualification-manifest-12.json` has SHA256
`4f0b0cbf0e06b7917405edc22bf852a31b037fce19361bde596841afda3c3f10`.

The native library pin is `6e3cbfb940be89e28d1d71c264fd8c3a4e44afeb`.
At that pin, `src/capi/vllm_c.cpp:993` lazily gets the asynchronous engine.
At `src/vllm/entrypoints/model_loader.cpp:2430-2444`, the accessor constructs AsyncLLM lazily.
Construction therefore happens on the first completion call.
At controller `4954b465`, `native_adapter.cpp:49` requires canonical prompt arrays before loading.
The adapter waits for the whole corpus before replying.
A missing aggregate therefore does not locate the fault within the corpus.

Historical `qualification-59a9c319-12` completed four native corpora on the same bound binary.
Those failures were token mismatches, not a GPU hang.
Its retained full result has SHA256
`1d518a475a79947d9921be92e9893ed74a4831359063284524b3c32a0db4f400`.
The original diagnostic-07 also started with c1, but lacks the current binding and lifecycle protections.
Do not reuse its old shell recipe as the new control.

## Scope and proposed design

Extend `tools/bench/strix_four_engine/lifecycle_diagnostic.py` from #3110.
Do not create another lifecycle controller or alter inference engine code.
The public `main(argv=None)` must reach the checked bootstrap and execution path.

Add explicit `--native-first-c1` mode.
Require `--canonical-reference` and `--canonical-reference-sha256` in this mode.
The reference is a retained full qualification result, not an unbound token list.
Use the existing explicit source archive, revision, hash, manifest, and output arguments.
Never compute an expected hash from the same input being checked.

The mode selects exactly one engine: `vllm.cpp`.
Reject conflicting, duplicate, or additional explicit engine selections before launch.
Keep ordinary c4 diagnostic behavior unchanged.
Do not change the six prompts, 128 tokens, greedy sampling, four-slot baseline, or engine environment.
This is not the #3107 96-request workload.

Verify the reference hash before parsing and again after execution.
Require integer schema 1 at the reference's top level.
Require the reference `.manifest` to equal the supplied checked manifest.
This binds its model revision and complete model-file map, including tokenizer files.
Require `.manifest.engines["vLLM"].source_revision` to match the checked controller's oracle pin.
Require `.engines["vLLM"].configuration` to contain integer schema 1, integer ID 1, and status `ok`.
Require its `prompt_ids` to equal the reference's top-level `prompt_ids`.
Validate six nonempty integer token arrays with the checked controller's token bounds.
Recompute `workload_sha256` from the checked six prompt strings, validated IDs,
`tokens: 128`, `concurrency: [1, 4]`, and the exact four-slot requested settings.
Require both the stored hash and vLLM's requested settings to match.
Schema 1 has no `.workload` field; do not invent one or require v2 fields.
The retained workload hash is `89e1f11abb1c2ac867c4709a29c1ebbb7fd8aca325db33dcdc18e6ff039c17d4`.
Greedy sampling and ignored EOS are established by the bound adapter source, not an absent reply field.
Use the checked controller's canonical prompts, token validation, and configuration values.
Refuse incompatible source versions instead of translating a v2 workload silently.
A reference's failed output-token gate does not authorize a tolerance.
The reference supplies previously observed input tokenization only.
Retain the reference identity and the consumed prompt arrays in the new result.

After existing binding checks, enter the existing lifecycle context and construct the existing Adapter.
Configure with the verified canonical token IDs.
Send exactly this first run command, with transport identity added by Adapter:

```json
{"command":"run","phase":"qualification","concurrency":1,"repetition":0}
```

Then close through the existing normal lifecycle path.
Do not send a c4 warmup, a second corpus, or another engine's configure command.
Retain each command and reply, including configuration.
Use a distinct first-c1 diagnostic identifier and label the run as qualification, not warmup.
Require the existing structural run validation for six complete 128-token requests.

Cap each configure and run exchange at 180 seconds, without increasing a stricter manifest timeout.
Retain the existing bounded shutdown and error cleanup.
Bound the entire resource-controller job to 15 minutes, including binding verification.
Do not describe the 180-second exchange cap as a whole-job deadline.
Timeout, abnormal exit, failed cleanup, changed bindings, or incomplete output means diagnostic FAIL.
Stop after a fault. Do not retry or reset the device automatically.
The operator retains kernel messages before and after the run.

Reuse #3110's finalization and immutable publication.
Publish each finalized record and the terminal aggregate once.
A diagnostic PASS means only successful execution, structure, teardown, and unchanged bindings.
It never means token parity, a repaired GPU fault, accepted throughput, or a reproduced speed advantage.

## Alternatives and interpretation

The ordinary qualification command starts vLLM before native.
It cannot isolate the native process without preceding GPU work.
The fixed c4 diagnostic changes the first inference shape.
An unchecked shell pipeline omits the current lifecycle and binding protections.
These alternatives do not answer the selected first-c1 question.

A standalone failure demonstrates reproducibility without the preceding vLLM corpus.
A successful run leaves sequencing and intermittent device state as hypotheses.
Neither result identifies the originating kernel.
Commit a separate causal experiment after this control if attribution remains unresolved.

## Tests and implementation sequence

Modify the diagnostic command, its existing focused tests, this specification's evidence, and the scoped usage example.
Do not modify `qualify.py`, native adapters, the publisher, or lifecycle implementation.
Read the reviewed #3110 interface before editing. Stop if that interface differs materially from this design.

- Write a public-CLI test for the new mode before implementation.
- Capture the missing-mode refusal as the initial red result.
- Use the existing source-bound CPU fixture and real immutable publisher.
- Assert exact configure, c1 qualification, and shutdown ordering.
- Assert the native process is the only selected engine.
- Assert canonical IDs reach the real Adapter boundary.
- Add independent missing, malformed, stale, and mismatched reference cases.
- Check model mismatch, changed reference bytes, wrong oracle pin, and incompatible workload refusal.
- Check conflicting engine selection before any launch.
- Check the 180-second cap and preservation of a stricter manifest timeout.
- Exercise configuration, run, timeout, abnormal-exit, and teardown failures without starting another engine.
- Preserve all #3110 publication and provenance regressions.
- Mutate the first-run concurrency, phase, ordering, reference guards, engine selection, deadline, and public call site independently.
- Require each mutation to fail for the intended reason and restore scratch bytes exactly.

Run the diagnostic, adopted-child, qualification, and adapter suites.
Run the pre-edit gate and the staged gate before committing.
Record the immutable full gate, exact range checks, and every omitted artifact check.
A fresh reviewer verifies the immutable change and its mutations.
The operator repeats the applicable gate before the single leased control.

## CPU implementation evidence

The implementation uses the checked schema-1 controller's `tokens`, `canonical`, `PROMPTS`, `RESOLVED`, and `PINS` interfaces.
Its workload recipe matches `qualify.py:341-342` at dependency `656ce703e6c48794c84d608733fbdba4c3e4461b`.
Token bounds match `qualify.py:257-260` at that revision.
The existing Adapter performs transport identity checks and shutdown.
The diagnostic records exchanges around that Adapter without modifying its transport or lifecycle.
The existing publisher finalizes each engine record and the aggregate exactly once.
No model artifact, adapter, engine, oracle pin, or acceptance rule changes.

Evidence lives in `verification-clean-3076.nWvRSX` under campaign `strix-four-engine-3053.X94a3J` on the resolved NAS.
CPU checks use the assigned `author.ext4` in a private mount namespace at literal `/tmp`.
The worker runs as `mudler` with a private `TMPDIR`. The wrapper unmounts the image on exit.

- `native3111-implementation-preedit.log`: full pre-edit preflight returned 0 at specification commit `0a5399ad09b49d5e7b4576daee698bd449b1b104`.
  SHA256: `913cfeae9a120b61c4a806c01d38532cece8e179cb8c58d99ca1e31911cfd55a`.
- `native3111-red.log`: the public-entry test failed before implementation because argparse refused the three new options.
  The test returned 1. The diagnostic returned 2.
  SHA256: `df3528464ad809230b8227126612954af6849128e3b679a7f6671a2452282704`.
- `native3111-mutations.json`: 34 independent mutations returned the intended assertion failure and restored the driver bytes exactly.
  SHA256: `420cd38355e1e3c32805cb38f0e5ea70a678761d0bc822be8300c0732fc28e82`.
  The mutation record includes each command's test selector, exit status, restoration hash, and log hash.
- `native3111-mutations-final.log`: the restored diagnostic suite passed 24 tests and returned 0.
- `native3111-focused.log`: the diagnostic, adopted-child, qualification, Python-adapter, and native-adapter suites passed 62 tests and returned 0.

Mutations cover public reachability, configure-before-run ordering, phase, concurrency, repetition, extra runs, engine selection,
deadlines, reference guards, retained evidence, and structural validation.
The real Adapter tests exercise configure, c1 qualification, shutdown, timeout, abnormal exit, and incomplete output.
Other CPU fixtures retain real source validation, lifecycle ownership, and immutable publication while replacing GPU engine work.

The preflight omits ARM ISA, CPU ISA, CUDA gencode, and Triton AOT artifact checks because their required build artifacts are absent.
Its argumentless PR-size check is separate from the explicit range classification required before handoff.
Staged and immutable preflight logs are retained with the same evidence prefix.
These CPU results establish no GPU execution result, fault cause, token parity, or performance advantage.

## Owed and stop conditions

#3111 owns the control implementation, review, hardware reproduction, and unresolved GPU fault.
#3110 owns the base diagnostic publication repair.
#3107 owns the expanded c1, c4, and c32 workload.
#3077 owns numerical diagnostic scoping and any separately ratified correctness proposal.
Strict token-exact benchmark acceptance remains binding.

Stop on missing canonical reference authority, changed pins, an incompatible controller, or unavailable device health evidence.
Stop if the control requires another engine launch, a reset, a tolerance, or modified inference scheduling.
Do not turn a successful control into a root-cause claim or close #3111 without resolving the fault.
