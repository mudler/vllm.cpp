# Donor evidence — #837 GetBlas dual-slot

Pinned **bytes**, not a dirty-tree HEAD. Implementation must copy this slice (or a later
immutable replacement that research re-reviews), not re-read `/home/don/llms/vllm.cpp`.

| Field | Value |
|---|---|
| Donor tree | `/home/don/llms/vllm.cpp` |
| Donor git HEAD | `2bb4bd8a` (dirty; this slice is **uncommitted** on that tree) |
| File | `src/vt/rocm/rocm_matmul_hipblaslt.hip` |
| Lines | 67–99 (`GetBlas`) |
| Slice | `getblas-fn-67-99.txt` |
| SHA256 | `9df2b163bc817db0d9545570136666c8e07a0bb600a01e50288a8f78c4148c51` |
| Recipient | `origin/main` `3ce5a1dc` `src/vt/rocm/rocm_matmul_hipblaslt.hip:72-99` (single `static thread_local Tls tls`) |
| Captured | 2026-08-14 |

`sha256sum` of the slice file must match the table. Do not treat `2bb4bd8a` as a clean
donor commit.
