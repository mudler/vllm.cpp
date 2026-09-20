# Isolate SGLang offload inspection

Row: `BACKEND-GATE-ROCM-SGLANG`. Issue: [#3074](https://github.com/mudler/vllm.cpp/issues/3074).

This is a prerequisite repair for [#3053](https://github.com/mudler/vllm.cpp/issues/3053).
Use a fresh implementer, an independent mutation reviewer, and operator verification.

## Evidence and source

Strix retry fb77c7b0 passed HIP identity, kernel compilation, installation, and the first common_ops import.
The next step, llvm-objdump --offloading, extracted host and HIP bundles beside the installed extension.
The extension-targets.log records the exact created paths under venv/site-packages/sgl_kernel.
Subsequent upstream test collection selected a common_ops.so.0.hipv4 bundle and failed to create an import spec.
The artifact directory is prepare-aab36a6a-rebuilt-03/logs in the #3053 NAS campaign directory.

At SGLang f63458b5beaceabbd9d749b9fc956370e1b649e6, load_utils.py:28-45 accepts filenames containing .so.
The fallback at lines 114-137 selects the first common_ops.* candidate and does not try every candidate after an import failure.
Our prepare.py invokes inspection on captured[extension] immediately before upstream tests.
Read the exact pinned loader and the observed extraction log before implementation.

## Design and scope

Only tools/bench/strix_four_engine/prepare.py, tests/tools/test_strix_four_engine.py, and this spec's evidence may change.
Preserve the #3072 HIP identity repair and all engine, model, toolkit, compiler, and compatibility-patch pins.
Do not modify the upstream loader, installed wheel contents, or the shared worker implementation.

Create an unused inspection directory beneath the session-local directory, outside the virtual environment and source tree.
Copy the captured installed extension into it without symlinks. Verify identical SHA256 values before inspection.
Invoke llvm-objdump --offloading with the copied operand and the inspection directory as cwd.
Keep the existing requirement that the observed architecture set is exactly gfx1151.
Record the installed extension hash, inspected-copy hash, source path, and inspection path in evidence.
Verify that inspection did not change the installed extension's bytes or package-file inventory.
An inspection error, copied-byte mismatch, or installation mutation must fail preparation without a success record.
Keep extracted bundles in the isolated diagnostic directory; never delete evidence to hide pollution.
Do not clean an already polluted installation and present it as a fresh preparation gate.

## Tests and gates

First extend the existing standard-library production CLI fixture with an inspector that writes adjacent offload bundle files.
The fixture's later import/test step must reject such files inside its installed common_ops package.
Run the original preparation code to an intended collection/import failure before implementing the repair.
After the repair, require the complete preparation CLI to pass and the installed package inventory to remain unchanged.
Assert that the inspected file is a byte-identical copy outside the environment and that diagnostic artifacts remain there.
Test wrong architecture, inspection failure, copied-byte mismatch, and unexpected installed-extension/package mutation independently.
Preserve all 17 current preparation tests. Do not replace an executable CLI test with source-string assertions.

The fresh reviewer mutates copied-operand selection, working-directory isolation, hash binding, installation-integrity checks,
and architecture rejection separately. Each mutation must fail the intended focused test. Restore all scratch bytes.
Run focused tests and the full applicable gate; report skipped artifact-dependent axes explicitly.
The operator repeats the gates on the reviewed head, then runs fresh preparation and all 1059 kernel tests on Strix under a lease.

## Stop conditions and owed work

A new legitimate upstream requirement needs source evidence and a design update before implementation.
No benchmark ratio can be accepted before model correctness. A kernel compilation pass is not a test or model-startup pass.
#3074 owns this isolation repair and the real collection/kernel gate. #3053 owns AITER, conversion, adapters, and measurements.

## Helper evidence

The helper started from committed design `a193ef98bcea832e25219fd1afbb3a47a9884cb1` on `row/BACKEND-GATE-ROCM-SGLANG-3074`.
The helper read the pinned archive's complete `sgl-kernel/python/sgl_kernel/load_utils.py` and the recorded extraction log.
The inspection now copies the extension into an unused session directory and verifies its SHA256 before invoking the inspector there.
The evidence records both paths, both hashes, and package inventories before and after inspection.
The package inventory includes file hashes, directories, and symlink targets.
Extracted bundles remain in the inspection directory for diagnosis.

Evidence directory: `/mnt/nas_share/rc/strix-four-engine-3053.X94a3J/offload-3074-helper`.
The focused command is `python3 -m unittest discover -s tests/tools -p test_strix_four_engine.py`.

- Red: append `-k test_cli_prepares_isolated_patched_rocm` to the focused command. It exited 1 before implementation.
  The fixture inspector created adjacent bundles. The later collection step rejected the polluted package. See `red.log`.
- Green: the focused command exited 0 with all 18 tests, including the previous 17 tests. See `green.log`.
  After scratch restoration, the same command exited 0 with 18 tests. See `restored.log`.
- Operand mutation: replace the copied operand with the installed extension. The success test exited 1 on installation mutation.
- Working-directory mutation: remove `cwd=scratch`. The success test exited 1 because the diagnostic bundle was outside the inspection directory.
- Copy-hash mutation: remove the mismatch guard. The inspection-failure test exited 1 because a corrupted copy was accepted.
- Integrity mutation: remove the installed hash and inventory comparison. The inspection-failure test exited 1 for three accepted installation mutations.
- Architecture mutation: remove the `gfx1151` guard. The inspection-failure test exited 1 because `gfx942` was accepted.
- Reachability mutation: replace the production call with an empty result. All six inspection-failure subcases failed because preparation accepted them.
  Each negative command used the focused command with `-k test_cli_refuses_inspection_failures`, except the first two success-test mutations.
  See `mutation-operand.log`, `mutation-cwd.log`, `mutation-hash.log`, `mutation-integrity.log`, `mutation-architecture.log`, and `mutation-reachability.log`.
  All mutations ran in a scratch copy. Its restored implementation SHA256 is `8cc34c2911dd3f997391a269dc6b01aa3b2e000e5eb8ba420740f22b4a324718`.
  One architecture attempt encountered insufficient host disk space. The private-mount rerun failed only the intended architecture subcase.
- Full preflight: `scripts/agent-preflight.sh --quiet` exited 0 with no failed gates. See `fullgate.log`.
  The helper ran as `mudler` with the evidence directory's separate 28 GiB `tmp.ext4` mounted privately on `/tmp`.
  The earlier host-filesystem preflight exited 1 under disk exhaustion and is not a passing result.
  Five gates reported `SKIP`: ARM ISA, CPU ISA, CUDA gencode, Triton AOT multiarch, and PR size.
  The first four require build artifacts outside this Python repair. PR size needs an explicit commit range and runs after commit.
- No helper GPU work ran. Real preparation, collection, and all 1059 kernel tests remain with the operator under a Strix lease.
  Model correctness and benchmark throughput, latency, and memory axes remain unmeasured by this repair.

## Inventory coverage repair

Independent review of `220bc0f714df604c99e6a22811a8ebb324dbc5bf` found three surviving inventory mutations.
The existing fixtures added files and symlinks but did not change existing entries or add an empty directory.
The repair adds three preparation CLI cases without changing production code.
The inspector changes existing `sgl_kernel.py` bytes, retargets an existing symlink, or creates an empty directory.
Each case requires preparation failure before upstream tests and refuses a success record.

Evidence directory: `/mnt/nas_share/rc/strix-four-engine-3053.X94a3J/identity-3072-gate.jZN4PR`.
The command is `python3 -m unittest discover -s tests/tools -p test_strix_four_engine.py`.

- `coverage-red-file.log`: replace inventory file hashes with a constant. The file-byte case fails on accepted preparation, exit 1.
- `coverage-red-symlink.log`: replace inventory symlink targets with a constant. The symlink-target case fails on accepted preparation, exit 1.
- `coverage-red-directory.log`: exclude directory entries. The directory case fails on accepted preparation, exit 1.
  Each mutation runs separately in a scratch copy with `-k test_cli_refuses_inspection_failures`.
  Each command reports exactly one failed subcase. The other eight refusal cases pass.
- `coverage-green.log`: the restored production command passes all 18 tests, exit 0.
  The scratch implementation and production implementation both hash to `8cc34c2911dd3f997391a269dc6b01aa3b2e000e5eb8ba420740f22b4a324718`.
  `cmp` confirms identical scratch restoration. Git confirms production bytes match the base commit.
- `coverage-final-gate.log` records `bash scripts/agent-preflight.sh --quiet` on the committed repair head.
  The final size, commit-style, and trailer checks use the explicit repair commit range.

Disk exhaustion prevented worktree creation under the cache directory.
The operator approved `/dev/shm/strix-offload-3074-coverage` and exclusive reuse of the released evidence directory's `tmp.ext4`.
The full gate uses that image as private `/tmp` and runs as `mudler`.
The initial host-filesystem preflight was stopped and is not a passing gate.
An initial symlink fixture failed before inspection because identity runs twice. The fixture now creates its link only when absent.
No GPU, oracle, model, or performance gate runs in this tests-only repair. The operator owns those gates.
