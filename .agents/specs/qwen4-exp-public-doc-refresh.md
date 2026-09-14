# Qwen3.8-Flash-Next public documentation refresh

## Now

Documentation audit at base `cef9f8216`, 14 September 2026. No model lifecycle,
runtime behavior, or benchmark acceptance changes. Implementation is complete.
Independent scoped review passes at `f450f7626`. The operator reran the focused
gates. Full preflight is incomplete because of the tool environment, as recorded
below. One pull request carries the spec and documentation commits.

## Scope

Replace the contradictory `Qwen4ExpForConditionalGeneration` entry in
`docs/FEATURES.md` with a concise account of current behavior. Add one README
news item for real-checkpoint text generation on CPU and ROCm. Preserve the
superseded feature cell verbatim in an era-stamped file under `.agents/completed/`.
Do not edit the model matrix, kernel code, benchmark numbers, or unrelated docs.
The existing EXL3 and HTTP multimodal documentation pull requests own those topics.

## Source and evidence inventory

| Surface | Authority | Required disposition |
|---|---|---|
| Production decode | `src/vllm/model_executor/models/qwen4_exp.cpp`, `qwen4_exp_forward.cpp`, and the model registry | Verify actual registration, decode routing, and one-sequence limit |
| Backend dispatch | `src/vt/rocm/rocm_qwen4_exp.hip`, quantized embedding registration and loader guards | Distinguish reachable code from measured real-checkpoint output |
| CPU checkpoint | `docs/bench-evidence/qwen4exp-released-checkpoint-tokens-20260831.md` | Prompt-dependent generation, no oracle token gate |
| ROCm checkpoint | `docs/bench-evidence/qwen4exp-rocm-hcnorm-gfx1151-20260913.md` | Generation on gfx1151, correctness and competitive speed still ungated |
| CUDA and artifacts | `docs/USAGE.md`, retained CUDA token evidence, owning model spec | No obsolete claim that CUDA emits no tokens; preserve actual unresolved token disagreement |
| Oracle | `.agents/oracles/vllm.md`, `llama-cpp-qwen4exp.md`, owning model spec | vLLM registration exists; no claim of a completed primary-oracle run |

Read exact paths and line anchors before writing. The public feature entry must
name the tested UD-IQ1_S GGUF, single-sequence serving, text-only artifact, and
unresolved correctness. Link evidence instead of repeating attempt history.
Do not publish an engine ratio or promote a liveness measurement to parity.

## Design and risks

Keep the existing feature-table columns. Use short sentences and descriptive
links. README news links to the feature entry or its evidence. Do not introduce
new commands: existing usage recipes retain authority. Archived relative links
must resolve from the archive, without changing the quoted source cell itself.

The main risk is mistaking a stale sentence for current behavior. Reconcile
conflicts against source and later committed evidence. Runtime verification is
unavailable on this CPU-only host; no new runtime claim is permitted.

## Tests and gates

This is a documentation correction, with no upstream test port or runtime
benchmark applicable. Capture the contradictory before text as the failing
baseline. Run existing `check-readme-structure.py`, `check-supported-models.py`,
`check-site.py`, `check-agent-record.py`, and their relevant mutation suites.
Run full preflight and distinguish unchanged baseline failures from regressions.
An independent reviewer checks every new claim against source and evidence,
checks preserved archive bytes, and mutates links in a scratch copy to verify
the link gate. The operator reruns the focused checks at the reviewed SHA.

## Completion and stop conditions

Done when scoped documentation is source-checked, focused checks pass, a fresh
review passes, and the fork pull request is open. No upstream merge is authorized.
Stop for a claim requiring a new GPU run; describe the gap without filling it.

## Owed

`ISSUE-LOCAL-01M2EXYPHDZ03EX01VRTB5FECQ` owns this documentation correction.
The model's existing runtime issues remain with its owning spec.

## Outcome

The feature row now describes the released text-only UD-IQ1_S GGUF and its
single-sequence limit. README news announces CPU and ROCm generation without
claiming oracle correctness or competitive performance. CUDA generates tokens,
but its disagreement with CPU remains unresolved. The original feature row is
preserved verbatim in `../completed/qwen4exp-features-history-20260914.md`.

Source checks at implementation base `12abed189`:

- `src/vllm/model_executor/models/qwen4_exp_registry.cpp:365` refuses multiple
  requests. Lines 1322 and 1327 declare the limit and register the model.
- `src/vt/rocm/rocm_ops.hip:207` registers quantized embedding on ROCm.
- `docs/bench-evidence/qwen4exp-released-checkpoint-tokens-20260831.md` records
  prompt-dependent CPU output and explicitly excludes an oracle token gate.
- `docs/bench-evidence/qwen4exp-rocm-hcnorm-gfx1151-20260913.md` records generation
  on gfx1151 and the missing correctness gate.
- `docs/bench-evidence/qwen4exp-cuda-decode-identifiers-20260902.md` records CUDA
  output and token disagreement. `docs/USAGE.md:1045` retains the unresolved
  disagreement and the artifact's text-only scope.
- `.agents/oracles/vllm.md:68` records the blocked model run.
  `.agents/oracles/llama-cpp-qwen4exp.md:198` records the scoped oracle's run.

## Verification evidence

The captured before row contains both “NO TOKEN HAS COME OUT OF A CUDA DEVICE”
and “A GPU HAS NOW PRODUCED TOKENS”. Its claim that no usable checkpoint token
exists contradicts its later CPU result. Source and retained runs falsify those
claims. A byte comparison against `git show 12abed189:docs/FEATURES.md` confirms
that the archive preserves the complete original row.

Commands use the temporary Python and Bash tools selected by the operator:
`PATH=/tmp/vllm-doc-tools/root/usr/bin:/tmp/vllm-doc-tools/root/bin:$PATH` and
`LD_LIBRARY_PATH=/tmp/vllm-doc-tools/root/usr/lib`.

| Command | Result |
|---|---|
| `python3 scripts/check-readme-structure.py` | PASS, exit 0 |
| `python3 scripts/check-supported-models.py` | PASS, exit 0 |
| `python3 scripts/check-site.py` | PASS, exit 0 |
| `python3 scripts/check-agent-record.py` | PASS, exit 0 |
| `python3 -m unittest discover -s tests/scripts -p 'test_check_readme_structure.py'` | PASS, 19 cases |
| `python3 -m unittest discover -s tests/scripts -p 'test_check_supported_models.py'` | PASS, 11 cases |
| `python3 -m unittest discover -s tests/scripts -p 'test_check_site.py'` | PASS, 7 cases after the operator installed Hugo. The initial missing-Hugo error also occurred on the untouched base |
| `python3 /tmp/verify-qwen4exp-doc-links.py "$PWD"` | PASS, all 7 changed links resolve, including both heading fragments |
| `git diff --check` | PASS, exit 0 |

The temporary link verifier reads the new README item and feature row. It checks
each relative path and compares each fragment with the target's Markdown
headings. In a scratch directory, replacing the README target with
`docs/MISSING.md#registered-architectures` fails with exit 1. Replacing its
fragment with `#missing-anchor` also fails with exit 1. Restoring the original
README passes. The source tree stays unchanged throughout both mutations.

Independent review of `f450f7626` found no issues. All seven changed link paths
and both fragments failed when corrupted in a scratch copy. Each byte-exact
restoration passed. The operator independently reran the four focused checks,
all 37 mutation cases, the link verifier, and the archive comparison.

Full `bash scripts/agent-preflight.sh` was attempted on the baseline,
implementation, and review trees. The logs are
`/tmp/vllm-doc-preflight-base.log`, `/tmp/vllm-doc-preflight-impl-base.log`, and
`/tmp/vllm-doc-preflight-review.log`. The baseline and implementation encountered
the same nine failing checks. The review encountered five of those failures.
The registration checker cannot find CMake. Release subprocesses also remove
the library path that the temporary Python installation needs. The operator
stopped the incomplete sweeps after recording these failures. Full preflight
is **PENDING**, not green. No GPU, runtime, or new benchmark gate applies to
this documentation-only correction.
