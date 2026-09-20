# Read the SGLang device limit through pinned HIP

Row: `BACKEND-GATE-ROCM-SGLANG`. Issue: [#3072](https://github.com/mudler/vllm.cpp/issues/3072).

This repairs the preparation prerequisite for [#3053](https://github.com/mudler/vllm.cpp/issues/3053).
Use a fresh implementer and an independent mutation reviewer. Preserve all parent benchmark pins and settings.

## Observed failure

Strix preparation job c13a24ee-d1ad-49a7-ba08-e96e814273c0 passed Torch import after the MIOpen repair.
Its identity probe then raised AttributeError for torch.cuda.get_device_properties(0).shared_memory_per_block.
The pinned AMD Torch 2.9.1 build does not expose that member.
Evidence: prepare-c8d019447-rebuilt-02/logs/identity-dependencies.log under the #3053 artifact directory.

## Source and design

The pinned Triton wheel is triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl.
Its triton/backends/amd/driver.c:199-209 calls hipGetDeviceProperties and returns props.sharedMemPerBlock as max_shared_mem.
Its driver.py:161-177 exposes get_device_properties; lines 700-705 use that method to identify the active device.
Read these exact wheel sources before implementation. The operator verifies the executing path on Strix.

Change the production identity probe to obtain the measured shared-memory limit through the pinned active Triton driver.
Require the AMD/HIP backend and retain the gfx1151, HIP, Torch, Triton, isolated-import, and distribution-count checks.
Record the property source and returned value. Never substitute the minimum required value as a measured result.
Retain the 65536-byte minimum and reject missing, non-integer, Boolean, negative, and insufficient values.
Propagate driver-query errors as failed preparation. Do not silently fall back to a constant or a different package.
Do not change toolkit, compiler, Torch, Triton, SGLang, compatibility patch, or model pins.

## Scope and tests

Only tools/bench/strix_four_engine/prepare.py, its focused test file, and this spec's evidence are implementation scope.
Work from this committed design in a separate linked worktree. Do not repair in the coordinator or shared checkout.

First add a standard-library regression that executes the real IDENTITY probe body.
Supply test modules whose Torch device properties have gcnArchName but no shared_memory_per_block member.
Supply a fake active HIP driver's measured property dictionary, not a canned successful identity JSON result.
Require the current code to fail for the missing Torch member before changing implementation.
After the repair, require a valid 65536-byte device to pass and its recorded source/value to match the query.
Add independent tests for missing, invalid, Boolean, negative, insufficient, non-HIP, and failing driver responses.
Preserve all existing preparation tests and runtime identity guarantees.

Run focused tests and the full applicable gate. Freeze the implementation for a fresh review.
The reviewer mutates the query call, backend check, returned-value binding, type check, and threshold independently.
The tests must detect every mutation. Restore scratch files byte-for-byte and report all skipped gate axes.
The operator reruns the focused and full gates, then retries the reviewed preparation on Strix under a lease.

## Stop conditions and owed work

A real driver-query failure or disagreement between HIP device identity and Torch requires investigation before acceptance.
No GPU build or benchmark runs outside a lease. No preparation success or throughput claim precedes its actual gate.
#3072 owns this repair and its real identity validation. #3053 owns kernel qualification, AITER startup, and the four-engine benchmark.

## Implementation evidence

The helper started from committed design `3d8bc2736` on `row/BACKEND-GATE-ROCM-SGLANG-identity`.
The focused test file is `tests/tools/test_strix_four_engine.py`.
The helper read the pinned wheel's AMD `driver.c:199-209`, `driver.py:161-177`, and `driver.py:700-705` before implementation.
The repair queries device 0 through the active HIP driver and checks agreement with Torch's architecture.
The identity record stores the returned limit and its property source.
The 65536-byte floor stays unchanged because the compatibility patch requires that device limit.

Evidence directory: `/mnt/nas_share/rc/strix-four-engine-3053.X94a3J/identity-3072-gate.jZN4PR`.

- Red: `python3 -m unittest discover -s tests/tools -p test_strix_four_engine.py -k reads_hip_limit` failed.
  Both subcases raised the intended `AttributeError` for the absent Torch member. See `red.log`.
- Green: `python3 -m unittest discover -s tests/tools -p test_strix_four_engine.py` exited 0 with 17 tests.
  See `focused.log`. The probe tests execute the production `IDENTITY` body through `identity()`.
- Mutations: `python3 /home/mudler/.cache/identity-3072-mutations.py` exited 0. See `mutations.log`.
  Eight independent source mutations broke the query, backend guard, returned value, integer check, floor, architecture guard, source record, and query-error propagation.
  Every mutation failed its focused tests. The runner mutates an isolated module in memory and leaves the file bytes unchanged.
- Full preflight: `scripts/agent-preflight.sh --quiet` exited 0 with no failed gates.
  See `full-preflight.log` and `gate-runner.sh` for the private mount recipe.
  The runner used a 28 GiB ext4 image for temporary files and ran as `mudler`.
  Five gates reported `SKIP`, so the runner correctly withheld an all-green verdict.
  Four need build artifacts outside this Python repair: ARM ISA, CPU ISA, CUDA gencode, and Triton AOT multiarch.
  The fifth is `check-pr-size.py`, which requires an explicit commit range and runs separately after the commit.
- The helper did not run GPU work. Real HIP validation remains with the operator under a lease.
  Model correctness, kernel qualification, throughput, latency, and memory measurements remain outside this repair.
