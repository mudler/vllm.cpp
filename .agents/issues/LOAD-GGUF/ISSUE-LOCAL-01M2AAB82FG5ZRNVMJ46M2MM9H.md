ID: ISSUE-LOCAL-01M2AAB82FG5ZRNVMJ46M2MM9H
Title: The GGUF load path never registers the load_stats reporter, so VT_LOAD_STATS's device_upload counter is unreachable on exactly the lane that needs it
Row: LOAD-GGUF
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

`ReportLoadBytes()` (`src/vllm/entrypoints/model_loader.cpp:445`) does two things: it prints the `bytes@load-end` line, and it registers the `std::atexit` handler that prints `bytes@exit`. The comment on it says why the second matters — "the device uploads are LAZY: ResidentWeight runs at first forward use, after this function returns, so the load-end snapshot always reads device_upload=0. Print the final totals at exit as well, which is where the 'bytes moved by this process' question is actually answered."

It is called at `:3319` and `:3360`, both on the safetensors branch. The GGUF branch calls `ReportGgufLoadIo()` (`:3103`) instead, which reports prefault spans and paged-in bytes and nothing else. So on a GGUF load NOTHING registers the atexit handler and `device_upload_bytes` is never printed at all.

The substitution was deliberate and its reason is recorded at `:425-429`: on a GGUF load `host_copy` and `borrowed` are both structurally zero, because `AddHostCopy` and `AddBorrowed` are only ever called on the safetensors path, and "a zero that means 'nobody counted' printed beside two real timings is worse than no line, because it reads as a measurement". That reasoning is right about two of the three counters and wrong about the third. `AddDeviceUpload` is called from `ResidentWeight` in `dense_attn_block.h:240` and `qwen3_5.cpp:1297`, which are loader-agnostic: they run on whatever weights the forward touches, GGUF included. Suppressing the whole line to hide two honest zeros also suppressed the one counter that is live on that lane.

This blocks a measurement in flight. `.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2AA9C31GCSDV8NRW26GMEVS.md` needs the per-step device-upload delta to decide whether the per-step MoE adapter rebuild re-uploads the expert towers or merely re-tests an aligned pointer, and the artifact in question is a GGUF.

The fix is to report the counters the GGUF lane actually keeps, rather than all three or none: print `device_upload` for GGUF and register the same atexit handler, leaving `host_copy` and `borrowed` off that line so no zero reads as a measurement.

## Resolution

-

### Why this was filed and not fixed in the same flow (2026-09-12)

AGENTS.md's in-flow rule is "file it, fix it in the same flow", and it exempts a
fix that needs its own path. This one does, for a reason that is about the gate
and not about the size of the diff. The defect is REACHABILITY — the branch does
not call the reporter — and per `.agents/reachability.md` the gate for that is
deleting the production call site and watching a focused gate go red. No test in
this tree sets `VT_LOAD_STATS` at all (`grep -rln VT_LOAD_STATS tests/` is
empty), and there is no harness that loads a GGUF and reads the process's
stderr, so the gate that would hold the fix does not exist yet. A unit test over
an extracted "which counters does this lane report" helper would pass whether or
not the GGUF branch ever calls it, which is the exact shape of green this
repository has been burned by.

So the honest sequence is: a row that builds the e2e surface, then the one-line
wiring behind it. Recorded here rather than left as a silent deferral.

The measurement that prompted this is NOT blocked on it. An `nsys` trace over a
bracketed decode window reports `cudaMemcpy`/`cudaMemcpyAsync` byte totals
directly, which answers the per-step re-upload question better than a cumulative
process counter would.
