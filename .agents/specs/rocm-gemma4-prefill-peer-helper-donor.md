# Donor evidence — #839 prefill peer helper

Pinned **bytes**, not a dirty-tree HEAD. Implementation may use these slices as the
Launch/Finish / PeerPipe / cache **structure** donor. Product **must not** copy the
donor unpin-before-`ev_e` lifetime (research `64cb` stop-ship 3). See the spec.

The slices sit beside the spec as `.log`, not under `.agents/evidence/`, because
`scripts/check-pr-size.py` classifies `.agents/specs/*.md` (`SPEC`) and
`.agents/specs/*.{patch,json,log}` (`SPEC_EVIDENCE`) and nothing else in
`.agents/`; an unclassified path makes that required check refuse the whole
change rather than the file. `.txt` is unclassified too, so the extension is
`.log`. Renaming does not touch the bytes and every SHA256 below is unchanged.

| Field | Value |
|---|---|
| Donor tree | `/home/don/llms/vllm.cpp` |
| Donor git HEAD | `2bb4bd8a` (dirty; these slices are **uncommitted** on that tree) |
| File | `src/vt/rocm/rocm_gemma4_experts.hip` |
| `rocm-gemma4-prefill-peer-helper-donor-peerpipe-40-58.log` | lines 40–58 SHA256 `a40291b8db72fdbf41767516095ba358488afb92dbbb6cf90bf8c11697a3c005` |
| `rocm-gemma4-prefill-peer-helper-donor-dequant-cache-core-60-209.log` | lines 60–209 SHA256 `cdb4da131c5d4786bafac7b8f0601362c254409455773258961636ddfc43b937` (`PrefillDequantCacheSlots`, `Slot`, `FreeAll`, `Ensure`, `GetLocked`) |
| `rocm-gemma4-prefill-peer-helper-donor-dequant-cache-210-232.log` | lines 210–232 SHA256 `16de816ee053e9a0ae22d6eefb7ada9c811d99196f81a61047e5ac71796447c9` |
| `rocm-gemma4-prefill-peer-helper-donor-wrapper-950-1100.log` | lines 950–1100 SHA256 `e8c54fe8f415ca277bd5185f9e856dc98428030e241d6da819cd72cad107e098` |
| `rocm-gemma4-prefill-peer-helper-donor-launch-finish-1090-1310.log` | lines 1090–1310 SHA256 `ae1828d721e50a18b099bca4b08fc79cdc595410cf8d179f1f18744e0f6c1095` |
| Recipient | `origin/main` `3ce5a1dc` `rocm_gemma4_experts.hip:648` (monolithic) |
| Hanging compare | `vllm.cpp-bc64fa-r2` `1b1baf43` `:692` |
| Captured | 2026-08-14 |

Known donor lifetime holes (do not productize as-is):

- Launch unpins at ~1262–1265 immediately after enqueueing GEMMs and **before** `ev_e`;
  Finish ~1274–1300 only enqueues `hipStreamWaitEvent` + copy and does **not** host-wait.
- Cache is process-wide (`DequantCacheSlotFor`, lines 220–231).
- `Ensure` (core 60–209) calls `FreeAll` on any device/I/H/nslots change; `FreeAll`
  hipFrees and zeros **pinned** slots. Product must reject/defer reconfiguration
  while any `pins > 0`.

`sha256sum` of each slice file must match the table. Do not treat `2bb4bd8a` as a clean
donor commit.
