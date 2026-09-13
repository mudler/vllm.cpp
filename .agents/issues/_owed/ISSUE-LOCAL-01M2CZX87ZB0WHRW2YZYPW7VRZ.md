ID: ISSUE-LOCAL-01M2CZX87ZB0WHRW2YZYPW7VRZ
Title: The variadic harness never restores its cached vllm-server, because the CIFS share is mounted file_mode=0664 and the cache guard tests -x
Row: -
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

benchmarks/variadic/job.sh:183 guards the binary cache with [ -x "$CACHED_BIN" ]. The rc share that holds the cache (/workspace on dgx:gpu0) is mounted cifs with file_mode=0664 and nounix, so no file on it is ever executable and the guard never holds. Measured 2026-09-13: /workspace/exl3-3150-ab/head/bin/39d3af455866bc47d76b0e0bd27fc3693e9e6ff5/vllm-server exists and reports -x FALSE, and the resubmitted #3150 A/B rebuilt that same pin from source (08:14:39 to 08:40:52 UTC, 26 minutes of a 35-minute setup) after a crash. dgx:gpu0 crashes under load about hourly, so the harness is resumable by design, and this guard throws away most of each boot's window: that boot crashed at 08:52, with 8 minutes of legs. The comment at line 226 names the cached binary as the resume guard, so the defect defeats the property it states. The restore branch already chmods the local copy (line 190), so an existence test is sufficient.

## Resolution

2026-09-13: fixed on row/BENCH-QWEN38-EXL3-VARIADIC-CACHE (73e378b06, fd6460123). The guard tests existence, the cache write is .so files, then MANIFEST.md5 renamed into place, then the binary renamed into place, and a restore verifies md5sum -c --strict on the local copies and rebuilds on any mismatch. tests/scripts/test_variadic_harness.py BinaryCacheRestores runs the real job.sh fragments, 7 cases; a fresh re-review returned PASS with M1-M6 mutations, and the operator reran the 35-test harness suite green. Open follow-up, LOW: the cache key is the source pin only, so a build-flag change under an unchanged pin restores the older build; the restore does not log the recipe.
