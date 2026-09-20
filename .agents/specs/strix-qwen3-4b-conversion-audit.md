# Qwen3-4B conversion audit implementation plan

Row: `BACKEND-GATE-ROCM-SGLANG`. Issue: [#3053](https://github.com/mudler/vllm.cpp/issues/3053).

> For agentic workers: use a fresh implementer and an independent mutation reviewer.
> Follow the repository's spec-driven execution contract. Do not run GPU work outside the operator's lease.

**Goal:** Prove that the llama.cpp BF16 GGUF contains the same tensor values as the pinned native checkpoint.

**Architecture:** Read the source safetensors and converted GGUF independently. Map tensor names explicitly, compare logical shapes and exact decoded values, and reject unexplained tensors. Emit an audit record that later qualification must verify.

**Tech stack:** Python, NumPy, safetensors, and the GGUF reader from pinned llama.cpp.

**Spec:** [Matched four-engine benchmark](strix-four-engine-qwen3-4b.md), Task 2.

## Global constraints

This supplement changes no model, engine pin, precision, workload, or correctness gate.
Model: `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`.
Converter: stock llama.cpp `10bf611e533d81f739128304991c5e133c6aebd8`, b10451.
Use `--outtype bf16`; never use an unrelated preconverted GGUF.
Keep existing evidence and refuse an existing output path. The operator stages weights locally and invokes conversion.
This audit is CPU work. It does not establish model gateability or authorize performance ratios.
Implementation starts after Task 1's reviewed preparation code is available; model qualification still requires its real Strix gate.

## Now

The CPU implementation checks synthetic real-format fixtures. Independent review and the real checkpoint audit remain pending.
This result establishes neither model gateability nor a performance ratio.

## Source anchors

- `conversion/qwen.py:52,67,155` at the converter pin defines Qwen tensor transforms and Qwen3 registration.
- `conversion/base.py:614,623,915,934,998` controls mapping, shapes, dtype promotion, and writing.
- `gguf-py/gguf/tensor_mapping.py` supplies GGUF names; do not replace an explicit Qwen mapping with permissive suffix matching.
- `src/models/qwen3.cpp:22-27` permits a missing output tensor and aliases tied embeddings.
- The source checkpoint index contains 398 tensors: two global tensors and eleven tensors per layer across 36 layers.

Read these exact pinned paths before implementation. Use Git history for any discrepancy; do not infer a new model layout.

## Task 2a: audit exact conversion {id: 2a, deps: [1]}

**Files:** Create `tools/bench/strix_four_engine/audit.py`,
`tests/bench/test_strix_four_engine_audit.py`, `tests/bench/requirements-strix-audit.txt`,
and `.github/workflows/strix-conversion-audit.yml`.
Do not edit model loaders, converter source, product headers, other engine code, or shared record files.

**Test dependencies:** Keep the standard-library-only `tests/tools` discovery unchanged.
The real-format suite belongs in `tests/bench` and has a dedicated required workflow with no path filter.
That workflow provisions an isolated Python 3.12 environment and checks out the exact converter revision above.
Pin NumPy 2.3.5 and safetensors 0.7.0, plus exact versions of the GGUF reader's transitive dependencies.
Use the pinned converter's `gguf-py` directory, not a separately published GGUF package.
Record the complete installation and execution commands in the workflow and this spec's evidence.
The suite accepts `STRIX_AUDIT_CONVERTER` only as an explicit source location and verifies its binding.
Missing dependencies or the converter are errors in this dedicated suite, never skipped tests.
Do not install packages or access the network from preflight or a checker.
The dedicated suite is an additional mandatory implementer, reviewer, and operator gate.
The workflow and its test invocation require static review and a mutation that removes the invocation.
This scope resolves the undeclared dependency in the first draft; it does not weaken the real-format fixture obligation.

**CLI:** `python audit.py --manifest INPUT.json --gguf MODEL.gguf --output UNUSED.json`.
The manifest supplies the model revision, local model directory, per-file SHA256 values,
and the converter source revision, directory, and source inventory digest.
The audit verifies all supplied hashes before reading and after comparison. Never trust a version string without source binding.
The GGUF reader must come from that pinned converter tree, not a different installed package.
Return zero only for an exact audit; return nonzero for any error, with a named failure and no passing record.
No downloading, conversion, installation, model execution, or subprocess configuration is hidden inside this CLI.

**Library:** `audit_model(manifest: dict, gguf_path: Path) -> dict` supplies the same verified result as the CLI.
Output binds model revision, source file hashes, GGUF SHA256, converter revision and inventory,
tensor mapping, logical shapes, source and destination storage types, element counts, and exact comparison results.
The top-level result is `PASS` only when all 398 expected source tensors are accounted for.
Record each exact-value F32 promotion explicitly. Keep measured counts derived from actual tensors, not copied as results from the expected inventory.

**Mapping:**

| Source name | GGUF name |
|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` |
| `model.norm.weight` | `output_norm.weight` |
| `model.layers.i.input_layernorm.weight` | `blk.i.attn_norm.weight` |
| `model.layers.i.post_attention_layernorm.weight` | `blk.i.ffn_norm.weight` |
| `model.layers.i.self_attn.q_proj.weight` | `blk.i.attn_q.weight` |
| `model.layers.i.self_attn.k_proj.weight` | `blk.i.attn_k.weight` |
| `model.layers.i.self_attn.v_proj.weight` | `blk.i.attn_v.weight` |
| `model.layers.i.self_attn.o_proj.weight` | `blk.i.attn_output.weight` |
| `model.layers.i.self_attn.q_norm.weight` | `blk.i.attn_q_norm.weight` |
| `model.layers.i.self_attn.k_norm.weight` | `blk.i.attn_k_norm.weight` |
| `model.layers.i.mlp.gate_proj.weight` | `blk.i.ffn_gate.weight` |
| `model.layers.i.mlp.up_proj.weight` | `blk.i.ffn_up.weight` |
| `model.layers.i.mlp.down_proj.weight` | `blk.i.ffn_down.weight` |

Indices `i` cover exactly 0 through 35. Require `tie_word_embeddings=true`, the expected architecture,
and the exact source inventory. No separate `lm_head.weight` or GGUF `output.weight` is expected here.
GGUF dimensions reverse the source logical order; reverse dimensions for shape comparison, not the tensor values.
No Q/K permutation applies to this conversion. Compare complete arrays, using bounded chunks if needed.
Require BF16 source tensors. Require BF16 GGUF matrices and exact F32 promotion for the one-dimensional tensors.
Reject NaN, infinity, duplicate names, aliases not specified above, quantized matrices, and tolerance-based equality.

- [ ] Write a failing CLI test using a complete synthetic 36-layer source inventory with small, distinctive BF16 values.
  Generate valid safetensors and GGUF fixture files through their format writers. Give different Q, K, and V values.
  The success test invokes the real CLI and checks every mapped tensor plus the source and output hashes.
- [ ] Add independent negative cases for one changed BF16 value; one changed promoted F32 value;
  a wrong shape; a Q/K permutation; a missing tensor; duplicate GGUF names; an extra output tensor;
  one unexpected source tensor; a wrong source dtype; a quantized destination; stale model/converter hashes;
  a wrong model or converter revision; non-finite values; and an existing output path.
- [ ] Run `STRIX_AUDIT_CONVERTER=<verified-source> <venv>/bin/python tests/bench/test_strix_four_engine_audit.py` and capture the intended red result.
- [ ] Implement the audit and rerun that exact command to focused green. Do not weaken expected tensors or equality.
- [ ] Run the repository's full applicable gate and commit the implementation with its reason and protocol trailers.
- [ ] Have a fresh reviewer mutate mapping, equality, each refusal, and the production CLI call site in scratch copies.
  Each mutation must fail the relevant focused test. The reviewer restores each scratch copy byte-for-byte.
- [ ] The operator reruns the focused and full gates, then runs the audit on the real converted checkpoint.

## Acceptance and stop conditions

The real audit must pass before llama.cpp participates in correctness qualification.
A smaller synthetic fixture verifies the harness, never the real 8 GB model.
An unexpected legitimate converter transform requires source evidence and an updated committed design before implementation.
Do not change the converter pin, permit lossy conversion, or bypass the audit to obtain a throughput number.

## Owed

- #3053 owns implementation, independent review, the real conversion audit, and the downstream four-engine qualification and measurements.

## Implementation evidence

The dedicated suite uses the stock GGUF writer and reader from the converter pin.
The complete source inventory contains 3425 regular files, excluding `.git` and `__pycache__` entries.
Its canonical JSON SHA256 is `ad7a105b10602373f7936b15fe9ff76c4ba349b982c985115e2520b9172e0ad7`.
The audit requires this independent anchor and the caller's matching inventory digest.
Rebinding a modified tree to itself does not authenticate the revision.
No published `gguf` package participates in the dedicated environment.

The local preparation and dedicated invocation are:

```sh
git -C /home/mudler/_git/llama.cpp worktree add --detach /home/mudler/.cache/strix3053-audit-llama 10bf611e533d81f739128304991c5e133c6aebd8
python3 -m venv /home/mudler/.cache/strix3053-audit-venv
/home/mudler/.cache/strix3053-audit-venv/bin/python -m pip install -r tests/bench/requirements-strix-audit.txt
/home/mudler/.cache/strix3053-audit-venv/bin/python -m pip check
STRIX_AUDIT_CONVERTER=/home/mudler/.cache/strix3053-audit-llama /home/mudler/.cache/strix3053-audit-venv/bin/python tests/bench/test_strix_four_engine_audit.py
```

Installation and `pip check` exited zero. The dedicated workflow repeats this provisioning with Python 3.12 and the exact source revision.
The original CLI red exited 1 because `audit.py` did not exist.
Its log is `/home/mudler/.cache/strix3053-audit-red-dedicated.log`.
An additional red caught a GGUF architecture mismatch that the first implementation accepted.
The corrected implementation checks both native and GGUF architecture declarations.

The dedicated suite contains five tests and 24 independent refusal cases.
The valid fixture checks every one of the 398 mappings, shapes, dtypes, element counts, and file bindings.
The library tests change source files, converter files, and the GGUF during comparison.
They also refuse an already imported reader outside the pinned source tree.
A corruption beyond the first comparison chunk prevents a prefix-only audit from passing.
The workflow test executes the actual CI command with an unavailable converter.
A removed invocation cannot produce the required setup failure.

Mutation logs use `/home/mudler/.cache/strix3053-audit-mutant-<name>.log`.
The mutations cover mapping, equality, finiteness, shape, rank, both dtypes, both inventories,
source hashes, converter hashes, the independent converter pin, revisions, both architecture checks,
final file checks, existing output preservation, reader provenance, complete chunk traversal,
the production CLI call, and the workflow call.
The pinned reader itself rejects duplicate GGUF names at `gguf-py/gguf/gguf_reader.py:329`.
Every mutation is restored before the dedicated suite and full preflight run again.

The full pre-commit invocation is `scripts/agent-preflight.sh --staged`.
It uses the existing private ext4 image for `/tmp`, because the ordinary temporary filesystem is below the tools suite's free-space floor.
The log is `/home/mudler/.cache/strix3053-audit-full.log`.
The helper handoff records its exit status and each argument-dependent skip.
The first startup attempt failed tools discovery when the draft test was under `tests/tools`.
The amended design moved that suite to `tests/bench` without changing ordinary discovery.

No real model audit, GPU execution, correctness qualification, or throughput measurement is claimed by these CPU tests.

## Review repair evidence

Review found that a failed final write could leave a valid `PASS` at the requested output path.
The new regression reproduced that defect for both write and close failures before implementation.
Its log is `/home/mudler/.cache/strix3053-audit-repair-red-output.log`.

The CLI writes, flushes, synchronizes, and closes a unique sibling staging file before publication.
Linux `renameat2` with `RENAME_NOREPLACE` publishes the record as the last fallible operation.
An existing or concurrently created output remains unchanged.
The implementation requires Linux, an exported `renameat2`, and filesystem support for its no-replace operation.
An unavailable operation produces `AUDIT_FAIL`, without a hardlink, symlink, or overwrite fallback.
File creation retains the previous `open("x")` permissions and process ownership.
This contract does not claim persistence across a power failure after the directory entry changes.

Staging names start with `.audit-pending-` and are never consumable audit outputs.
Write, close, and publication failures remove the staging file before returning failure.
If the filesystem also refuses cleanup, the CLI reports `AUDIT_PENDING_CLEANUP_FAIL` and leaves that unpublished file for diagnosis.
Only the explicitly requested final output path can qualify a model.
No cleanup operation follows successful publication, so cleanup cannot turn a published result into a failed command.

The dedicated suite now checks exact promotion annotations for all norms and no promotion for matrices.
Independent fixtures isolate unbound and unexpected shards, duplicate JSON keys, escaping source bindings, and converter symlinks.
They also isolate model type, layer count, incorrect shard placement, and a foreign reader submodule under a pinned root package.
All source hashes remain otherwise valid in these fixtures.
Publication tests exercise write failure, close failure, a concurrent output, an unsupported filesystem, an unavailable symbol, and a non-Linux platform.

Each of the ten surviving review mutations now fails its corresponding fixture.
Additional mutations remove cleanup, the no-replace flag, publication error handling, the platform guard, and the production publication call.
All fifteen mutations fail, and the source is restored byte-for-byte after each mutation.
Logs use `/home/mudler/.cache/strix3053-audit-repair-mutant-<name>.log`.
The restored dedicated gate uses the unchanged command in the implementation evidence.
Its log is `/home/mudler/.cache/strix3053-audit-repair-final-focused.log`.
The immutable-head full preflight log is `/home/mudler/.cache/strix3053-audit-repair-full.log`.
The helper handoff records observed exits and argument-dependent skips.
Independent review and the operator's real checkpoint audit remain required.

## Publication failure coverage

The real CLI now has isolated filesystem synchronization and cleanup failure fixtures.
Both fixtures compare all 398 synthetic tensors before the injected filesystem error.
A synchronization failure must return `AUDIT_FAIL`, leave no final output, and remove its staging file.
If staging-file removal also fails, the CLI must report `AUDIT_PENDING_CLEANUP_FAIL` and retain that unpublished file.
The retained file contains the complete audit record but cannot qualify a model at its temporary path.

Deleting `os.fsync` makes the new fixture fail because the command incorrectly returns zero.
Deleting the cleanup diagnostic makes the cleanup fixture fail because the named warning is absent.
The mutation logs are `/home/mudler/.cache/strix3053-audit-coverage-red-sync.log` and
`/home/mudler/.cache/strix3053-audit-coverage-red-cleanup.log`. Both commands exited 1.
The restored dedicated suite passed 10 tests in 23.931 seconds with exit 0.
Its log is `/home/mudler/.cache/strix3053-audit-coverage-green.log`.
Production `audit.py` remains byte-identical, with SHA256 `c03a755d1ceb8c087e08cfd1dbeb12557c227b345f41c01a7fce45a62797f4fd`.
The immutable-head preflight log is `/home/mudler/.cache/strix3053-audit-coverage-full.log`.
The helper handoff records its exit and argument-dependent skips. Independent review and operator verification remain required.
