# Repair the blocked-stream fixture

Row: `BACKEND-ROCM-RESIDUAL-NORM`. Issue: [#3103](https://github.com/mudler/vllm.cpp/issues/3103).
This repair starts from `94ac5d1742c540fc5cbc2084d0fb74fdfd5dab41` in a separate helper worktree.

The operator's [red capture](operator-red.log) fails the existing nonblocking-stream assertion before testing kernel dispatch.
`RocmBackend::CreateQueue` uses `hipStreamCreate`, whose stream flags are zero.
The fixture now creates and owns a stream with `hipStreamNonBlocking`.
Its outer device scope outlives the stream, operands, and callback.
The release guard drains queued work before destroying its atomic callback flag, including assertion and exception exits.
The flag assertion and output comparison remain unchanged.
Production queue creation and residual normalization code remain unchanged.

The [build receipt](build-receipt.json) records exact compiler and linker arguments, input hashes, and private output binaries.
The helper rebuilds the test and test main against private copies of the unchanged base archives.
The [build script](build-fixture.py) records that adaptation; this is a test relink, not a full product rebuild.
A scratch mutation replaces the real residual provider's launch stream with the default stream.
The operator must confirm that this mutation passes the stream-flag assertion and fails the output comparison.
No tracked product file or original archive changes during the mutation build.

The [keyed-record proof](keyed-record-proof.json) starts from the complete target version of `.agents/backend-matrix.md`.
It changes only the `BACKEND-PLATFORM` citation from `backend.h:22` to `backend.h:23`.
Every other byte remains unchanged.
The [red record gate](record-red.log) reports the stale class citation; the [corrected gate](record-green.log) passes.

The operator independently completes the pinned upstream export: 144 core cases, 60 IR RMS cases, and 60 IR add cases.
All 1,452 payload hashes are verified in the [export checks](upstream-export-checks.json).
The [native comparison](upstream-native.log) passes 528 CPU/ROCm executions and 22,576 assertions on the unchanged base implementation.
The original model token gate still differs at six positions in the L33/C2 tail.
This fixture repair makes no numerical or token-parity claim beyond those measured component results.

The repaired CPU binary passes all six component tests and 8,985 assertions, including all 264 upstream cases.
The [CPU receipt](cpu-focused-receipt.json) and [raw output](cpu-focused.log) retain the exact command and result.
The [manifest](manifest.json) seals the retained captures and commands.
The repaired hardware fixture, wrong-stream mutation, full staged preflight, fresh review, and operator rerun remain pending at this checkpoint.
