ID: ISSUE-LOCAL-01M2BZ5QK4XRETK48CXKSHKRDW
Title: OWED: chunked H2D through a pinned bounce buffer, llama.cpp's 4 x 64 MiB shape
Row: MODEL-MM-QWEN4-EXP
State: CLOSED
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

ResidentWeight stages every weight with one hipMemcpyAsync straight out of a PAGEABLE, file-backed mmap. llama.cpp does not: llama-model-loader.cpp uses a set of pinned host buffers (4 x 64 MiB) and copies the weight through them in chunks, so the driver never has to pin or stage an arbitrarily large pageable range itself. If the gfx1151 svm_range_set_attr stall survives the host-residency fixes in .agents/specs/rocm-host-residency-after-upload.md, the pageable file-backed source is itself the trigger and this is the next change. It is deliberately NOT built there: it touches every staged weight on every backend and needs its own measurement.

## Resolution

Built and measured 2026-09-13. RocmBackend::Copy now stages a host-to-device copy of 64 MiB or more, out of unregistered host storage, into device memory, on a non-capturing stream, through four pinned 64 MiB buffers with an event each -- llama.cpp's shape at 10bf611e5, src/llama-model-loader.cpp:1440 and :1449. Spec .agents/specs/rocm-chunked-pinned-h2d.md; decision and loop in the HIP-free include/vt/rocm/rocm_pinned_h2d.h, table-tested in tests/vt/test_rocm_pinned_h2d.cpp (9 cases / 84 assertions) and reached through Backend::Copy in tests/vt/test_backend_cross_device.cpp (61 cases / 84841 assertions on strix:gpu0, rc job e8bf3b66). It is the change that makes the row's model gate pass: on strix:gpu0 the 67.56 GiB Qwen3.8-Flash-Next UD-IQ1_S produced three generations of 32 tokens with the ring on, and the SAME BINARY with VT_ROCM_PINNED_H2D_MIB=0 wedged in svm_range_set_attr for 135 of 197 wchan samples and was killed at 1200 s (rc job 672093bc).
