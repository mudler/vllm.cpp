ID: ISSUE-LOCAL-01M2BZ5QK4XRETK48CXKSHKRDW
Title: OWED: chunked H2D through a pinned bounce buffer, llama.cpp's 4 x 64 MiB shape
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

ResidentWeight stages every weight with one hipMemcpyAsync straight out of a PAGEABLE, file-backed mmap. llama.cpp does not: llama-model-loader.cpp uses a set of pinned host buffers (4 x 64 MiB) and copies the weight through them in chunks, so the driver never has to pin or stage an arbitrarily large pageable range itself. If the gfx1151 svm_range_set_attr stall survives the host-residency fixes in .agents/specs/rocm-host-residency-after-upload.md, the pageable file-backed source is itself the trigger and this is the next change. It is deliberately NOT built there: it touches every staged weight on every backend and needs its own measurement.

## Resolution

-
