ID: ISSUE-LOCAL-01M2MKEKWH9J1WMVY0ZQVG4MKS
Title: APEX e2e: binary_ng 'Invalid subtile broadcast type' blocks generation
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-16
Updated: 2026-09-16
Closed: 2026-09-16

## Problem

The APEX-I-Nano 27B e2e probe crashes in tt-metal's binary_ng_device_operation with 'Invalid subtile broadcast type' (binary_ng_device_operation.cpp:255). This is the next blocker after the L1 overflow fix (#3206) got the probe past the first forward pass. The ROW_MAJOR doctrine comment in NormalizeDevF32Tile names this as a fatal class. Need to attribute which of our ops triggers binary_ng with incompatible broadcast geometry, then fix it.

## Resolution

Closed 2026-09-16. Root cause: `NormalizeDevF32Tile` (the L1 overflow fix from
#3206) preserves the shadow's native rank-2 geometry. Two call sites received
tensors at the shadow's native shape instead of the intended target shape,
causing `binary_ng` (`ttnn::multiply`) to see operands with the same numel but
incompatible 2D shapes.

Fix 1 — `RmsNormGatedKernel` (`tenstorrent_ops.cpp:~6739`): added
`CaptureSafeReshape(act, {rows, d})` between `sigmoid`/`silu` and the multiply.
`act` is TILE layout (NormalizeDevF32Tile's final `to_layout(TILE)`), so
`view_device` (tensor_ops.cpp:398) takes the TILE safe path — no page_size
rebuild, no data corruption.

Fix 2 — `AttnQkNormRopeGateKernel` (`tenstorrent_ops.cpp:~4624`): added
`CaptureSafeReshape(dev_k, {t*hkv, dh})` after `NormalizeDevF32Tile`. Same TILE
safe path applies.

Evidence: APEX e2e probe with `VT_TT_SLOT_TRACE=1` confirmed both crash sites
red-before (RmsNormGated: `dim a: 12288, dim b: 6144`; AttnQkNormRopeGate:
`dim a: 1024, dim b: 256`) and green-after (probe advances through all 64 trunk
blocks and enters decode). Full backend suite 77/77, 524,495 assertions.

The probe then hits a new, separate blocker: OOM from memory fragmentation in
`MatmulBTQuant` → `tilize_with_val_padding` (2.5 GB temp for the full lm_head
weight). That is a different issue, not a regression of this fix.
