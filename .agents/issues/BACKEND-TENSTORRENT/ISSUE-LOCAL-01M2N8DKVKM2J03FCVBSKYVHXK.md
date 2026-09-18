ID: ISSUE-LOCAL-01M2N8DKVKM2J03FCVBSKYVHXK
Title: APEX e2e: keep-quant decode OOM — chunk policy misses the TILE-tiling blowup
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-16
Updated: 2026-09-16
Closed: -

## Problem

The APEX 27B e2e decode dies in MatmulBTQuantGroupedKernel -> DecodeKeepQuantWordsF32 -> repair -> to_layout(TILE): the repair lambda tiles {B,16,16} product planes to {B,32,32}, a 4x blowup the 256 MiB plane budget does not count. The ceil(N/8) trace term forces 31,040-row chunks on the head [248320,5120]; the largest single TILE alloc is then 2.5 GB (318 MB/bank) against a 238 MB largest free block. An env-lever probe (VT_TT_KEEPQUANT_CHUNK_BYTES=64MiB, chunk 3,276) completed the first full e2e generation (exit 0, 64/64 tokens), proving the attribution. The default policy must account for tiling so the knob is not required.

## Resolution

-
