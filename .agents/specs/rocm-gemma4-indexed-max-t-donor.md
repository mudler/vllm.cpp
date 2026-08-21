# Donor evidence — #838 indexed-max-T

Pinned **bytes**, not a dirty-tree HEAD. Implementation must copy these slices (or a later
immutable replacement that research re-reviews), not re-read `/home/don/llms/vllm.cpp`.

Donor includes `kPrefillIndexedNoSyncMaxT` / `DEVICE_GROUP` neighbors. **Those are out of
scope for this row.** Only `kDecodeIndexedMaxT` (default 63, clamp [1,63]) and the
`indexed_ok_t` predicate **without** the nosync disjunct are in scope.

| Field | Value |
|---|---|
| Donor tree | `/home/don/llms/vllm.cpp` |
| Donor git HEAD | `2bb4bd8a` (dirty; these slices are **uncommitted** on that tree) |
| File | `src/vllm/model_executor/models/gemma4_moe.cpp` |
| Slice A | `rocm-gemma4-indexed-max-t-donor-gate-911-933.log` lines 911–933 SHA256 `d0d28f3d55ff7d526475c9a2a1d028792cc245ba4cc2b421ebf994fed9b96e59` |
| Slice B | `rocm-gemma4-indexed-max-t-donor-ok-1088-1110.log` lines 1088–1110 SHA256 `5509f3f77dcadd023ce73743e13a2a1a0237d8dd0a8e25a98a765430fc577bd0` |
| Recipient | `origin/main` `3ce5a1dc` `gemma4_moe.cpp:735` (`T == 1` only) / `:1345` serial |
| Captured | 2026-08-14 |

`sha256sum` of each slice file must match the table. Do not treat `2bb4bd8a` as a clean
donor commit.
